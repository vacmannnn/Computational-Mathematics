import argparse
import time
import numpy as np
from PIL import Image


def evaluate_k(m, n, N):
    original_size = m * n * 3
    for k in range(min(m, n) + 1, 1, -1):
        compressed_size = 3 * (m * k + k + k * n) * 8
        if original_size / compressed_size >= N:
            return k
    return min(m, n)


class standard:
    @staticmethod
    def compress(image_data, ratio):
        compressed_data = {}
        k = evaluate_k(image_data.shape[0], image_data.shape[1], ratio)
        for channel in range(3):
            channel_data = image_data[:, :, channel]
            U, s, VT = np.linalg.svd(channel_data, full_matrices=False)
            compressed_data[f"U{channel}"] = U[:, :k]
            compressed_data[f"s{channel}"] = s[:k]
            compressed_data[f"V{channel}"] = VT[:k, :]

        return compressed_data

    def decompress(self, compressed_data):
        original_shape = compressed_data["original_shape"]
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = (
                compressed_data[f"U{channel}"],
                compressed_data[f"s{channel}"],
                compressed_data[f"V{channel}"],
            )
            reconstucted_data[:, :, channel] = (
                (U @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
            )
        return reconstucted_data


class primitive:
    @staticmethod
    def power_method(A, num_iterations=1000):
        v = np.random.rand(A.shape[1])
        v = v / np.linalg.norm(v)

        for _ in range(num_iterations):
            Av = A @ v
            v_new = A.T @ Av
            v_new = v_new / np.linalg.norm(v_new)

            v = v_new

        sigma = np.linalg.norm(A @ v)
        return sigma, v

    def deflate(A, sigma, u, v):
        return A - sigma * np.outer(u, v)

    def compress(self, image_data, ratio):
        compressed_data = {}
        k = evaluate_k(image_data.shape[0], image_data.shape[1], ratio)
        for channel in range(3):
            channel_data = image_data[:, :, channel]
            U, S, V = [], [], []
            for _ in range(k):
                sigma, v = self.power_method(channel_data)
                u = channel_data @ v
                u /= np.linalg.norm(u)

                U.append(u)
                S.append(sigma)
                V.append(v)
                channel_data = self.deflate(channel_data, sigma, u, v)

            compressed_data[f"U{channel}"] = np.array(U)
            compressed_data[f"s{channel}"] = np.array(S)
            compressed_data[f"V{channel}"] = np.array(V)

        return compressed_data

    def decompress(compressed_data):
        original_shape = compressed_data["original_shape"]
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = (
                compressed_data[f"U{channel}"],
                compressed_data[f"s{channel}"],
                compressed_data[f"V{channel}"],
            )
            reconstucted_data[:, :, channel] = (
                (U.T @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
            )
        return reconstucted_data

    def decompress_image(input_file, output_file, algorithm):
        compressed_data = np.load(input_file + ".npz")

        reconstructed_data = algorithm.decompress(compressed_data)
        write_bmp(output_file, reconstructed_data)


class advanced:
    @staticmethod
    def compress(image_data, ratio) -> bytes:
        np.random.seed(0)
        u = np.zeros((image_data.shape[0], ratio))
        sigma = np.zeros(ratio)
        v = np.zeros((image_data.shape[1], ratio))

        time_bound = time.time() * 1000 + 1000
        while time.time() * 1000 < time_bound:
            q, _ = np.linalg.qr(np.dot(image_data, v))
            u = q[:, :ratio]
            q, r = np.linalg.qr(np.dot(image_data.T, u))
            v = q[:, :ratio]
            sigma = np.diag(r[:ratio, :ratio])
            if np.allclose(np.dot(image_data, v), np.dot(u, r[:ratio, :ratio]), 1e-8):
                break

        data = np.concatenate((u.ravel(), sigma, v.T.ravel()))
        return data.astype(np.float32).tobytes()

    def decompress(self, compressed_data):
        original_shape = compressed_data["original_shape"]
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = (
                compressed_data[f"U{channel}"],
                compressed_data[f"s{channel}"],
                compressed_data[f"V{channel}"],
            )
            reconstucted_data[:, :, channel] = (
                (U @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
            )
        return reconstucted_data


def sqr(a):
    return 0.0 if a == 0.0 else a * a


def sign(x):
    return 1 if x >= 0.0 else -1


def write_bmp(file_path, image_data):
    image = Image.fromarray(image_data.astype(np.uint8))
    image.save(file_path)


def main():
    parser = argparse.ArgumentParser(description="Utility to compress BMP images")

    parser.add_argument(
        "-c", "--compress", action="store_true", help="Compress the image"
    )
    parser.add_argument(
        "-d", "--decompress", action="store_true", help="Decompress the image"
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-r", "--ratio", type=int, default=2, help="Compression ratio (default: 10)"
    )
    parser.add_argument(
        "-a", "--algorithm", choices=["standard", "primitive", "advanced"]
    )

    args = parser.parse_args()

    if (args.compress and args.decompress) or (
        not args.compress and not args.decompress
    ):
        print("ну что-то как-то так себе")
        return

    if args.compress:
        with Image.open(args.input) as image:
            sample = np.array(image)

        if args.algorithm == "standard":
            method = standard()
        elif args.algorithm == "primitive":
            method = primitive()
        else:
            method = advanced()
        compressed_data = method.compress(sample, args.ratio)
        compressed_data["original_shape"] = sample.shape
        np.savez(args.output, **compressed_data)

    if args.decompress:

        if args.algorithm == "standard":
            decomp = standard()
        elif args.algorithm == "primitive":
            decomp = primitive()
        else:
            decomp = advanced()

        compressed_data = np.load(args.input + ".npz")

        reconstructed_data = decomp.decompress(compressed_data)
        write_bmp(args.output, reconstructed_data)


if __name__ == "__main__":
    main()
