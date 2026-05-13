from pathlib import Path


def patch_botsort_threshold():
    file_path = Path(
        r".venv\Lib\site-packages\boxmot\trackers\botsort\botsort.py"
    )

    old = (
        "matches, u_unconfirmed, u_detection = "
        "linear_assignment(dists, thresh=0.7)"
    )

    new = (
        "matches, u_unconfirmed, u_detection = "
        "linear_assignment(dists, thresh=0.9) # patch fixed from 0.7 to 0.9"
    )

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    content = file_path.read_text(encoding="utf-8")

    if new in content:
        print("Already patched.")
        return

    if old not in content:
        raise ValueError("Target line not found.")

    content = content.replace(old, new)

    file_path.write_text(content, encoding="utf-8")

    print("Patch applied successfully.")


if __name__ == "__main__":
    patch_botsort_threshold()