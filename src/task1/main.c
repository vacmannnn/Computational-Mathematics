#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 64
#define EPS 0.1

struct net {
    size_t sz;
    double h;

    double **u;
    double **f;
};

typedef double (*fun_xy)(double, double);

static int min(int x, int y) {
    if (x < y) {
        return x;
    }
    return y;
}

double **create_arr(size_t sz) {
    double **res = calloc(sz, sizeof(*res));
    for (int i = 0; i < sz; i++)
        res[i] = calloc(sz, sizeof(*res[i]));
    return res;
}

void free_arr(double **arr, size_t sz) {
    for (int i = 0; i < sz; i++)
        free(arr[i]);
    return free(arr);
}

struct net *fill_net(size_t sz, fun_xy f, fun_xy u) {
    struct net *res = malloc(sizeof(*res));
    res->sz = sz;
    res->h = 1.0 / (sz - 1);
    res->f = create_arr(sz);
    res->u = create_arr(sz);
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            if ((i == 0) || (j == 0) || (i == (sz - 1)) || (j == (sz - 1))) {
                res->u[i][j] = u(i * res->h, j * res->h);
            } else {
                res->u[i][j] = 0;
            }
            res->f[i][j] = f(i * res->h, j * res->h);

        }
    }
    return res;
}

void clear_net(struct net *nt) {
    free_arr(nt->u, nt->sz);
    free_arr(nt->f, nt->sz);
    return free(nt);
}

static double block(struct net *nt, int a, int b) {
    int i0 = 1 + a * BLOCKSIZE;
    int im = min(i0 + BLOCKSIZE, nt->sz - 1);
    int j0 = 1 + b * BLOCKSIZE;
    int jm = min(j0 + BLOCKSIZE, nt->sz - 1);

    double dm = 0;
    for (int i = i0; i < im; i++) {
        for (int j = j0; j < jm; j++) {
            double temp = nt->u[i][j];
            nt->u[i][j] = 0.25 * (nt->u[i - 1][j] + nt->u[i + 1][j] + nt->u[i][j - 1] + nt->u[i][j + 1] -
                                  nt->h * nt->h * nt->f[i][j]);
            double d = fabs(temp - nt->u[i][j]);
            if (dm < d)
                dm = d;
        }
    }
    return dm;
}

size_t process_net(struct net *nt) {
    size_t iter = 0;
    size_t work_sz = nt->sz - 2;
    int numb_block = work_sz / BLOCKSIZE;
    if (BLOCKSIZE * numb_block != work_sz)
        numb_block++;
    double dmax;
    double *dm = calloc(numb_block, sizeof(*dm));

    do {
        dmax = 0;
        for (int nx = 0; nx < numb_block; nx++) {
            dm[nx] = 0;
            int i, j;
            double d;

#pragma omp parallel for shared(nt, nx, dm) private(i, j, d)
            for (i = 0; i < nx + 1; i++) {
                j = nx - i;
                d = block(nt, i, j);
                if (dm[i] < d) dm[i] = d;
            }
        }
        for (int nx = numb_block - 2; nx > -1; nx--) {
            int i, j;
            double d;

#pragma omp parallel for shared(nt, nx, dm) private(i, j, d)
            for (i = numb_block - nx - 1; i < numb_block; i++) {
                j = numb_block + ((numb_block - 2) - nx) - i;
                d = block(nt, i, j);
                if (dm[i] < d) dm[i] = d;
            }
        }
        for (int i = 0; i < numb_block; i++)
            if (dmax < dm[i]) dmax = dm[i];
        iter++;
    } while (dmax > EPS);
    free(dm);
    return iter;
}

double x4_p_y3(double x, double y) {
    return 52 * pow(x, 4) + 25 * pow(y, 3);
}

double dx4_p_y3(double x, double y) {
    return 624 * pow(x, 2) + 150 * y;
}

double rev_sinx_p_cosy(double x, double y) {
    return 1 / (sin(x) + cos(y));
}

double drev_sinx_p_cosy(double x, double y) {
    return 2 * cos(x) * cos(x) / pow(sin(x) + cos(y), 3) + sin(x) / pow((sin(x) + cos(y)), 2);
}

double sinx_p_cosy(double x, double y) {
    return 1 / 5 * sin(10 * x) + 1 / 5 * cos(10 * y);
}

double dsinx_p_cosy(double x, double y) {
    return -20 * (sin(10 * x));
}

double d_kx3_p_2ky3(double x, double y) { return 6000 * x + 12000 * y; }

double kx3_p_2ky3(double x, double y) { return 1000 * pow(x, 3) + 2000 * pow(y, 3); }

int main() {
    const int thread_nums[] = {1, 4, 8};
    fun_xy funcs[] = {x4_p_y3, kx3_p_2ky3, rev_sinx_p_cosy, sinx_p_cosy};
    fun_xy du_arr[] = {dx4_p_y3, d_kx3_p_2ky3, drev_sinx_p_cosy, dsinx_p_cosy};
    int nets[] = {20, 50, 100, 150, 250};

    for (int tn = 0; tn < 3; tn++) {
        omp_set_num_threads(thread_nums[tn]);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                struct net *net;
                net = fill_net(nets[j], funcs[i], du_arr[i]);
                double start_time = omp_get_wtime();
                size_t iters = process_net(net);
                double end_time = omp_get_wtime();
                printf("%f; %zu iterations; net size -- %d; threads -- %d; func id -- %d\n", end_time - start_time,
                       iters, nets[j],
                       thread_nums[tn], i);
            }
            printf("\n\n\n");
        }
    }
    return 0;
}