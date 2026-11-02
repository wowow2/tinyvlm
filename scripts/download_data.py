from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path("data/input.txt")


def main():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"{DEST} already exists, skipping")
        return
    urllib.request.urlretrieve(URL, DEST)
    print(f"saved {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()