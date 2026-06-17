import json
import os
import unicodedata

SRC = os.path.join(os.path.dirname(__file__), "..", "LSCS45", "sample.json")
OUT = os.path.join(os.path.dirname(__file__), "static", "signs")


def slug(name):
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    index = {}
    for signer, cats in data.items():
        for cat, signs in cats.items():
            for sign, vids in signs.items():
                vid = vids[sorted(vids.keys())[0]]
                rep = vid[sorted(vid.keys())[0]]
                frames = []
                for fk in sorted(rep.keys(), key=lambda k: int(k.split("_")[1])):
                    lm = rep[fk]
                    frames.append({
                        "pose": [lm["pose"]["x"], lm["pose"]["y"], lm["pose"]["z"]],
                        "l_hand": [lm["l_hand"]["x"], lm["l_hand"]["y"], lm["l_hand"]["z"]],
                        "r_hand": [lm["r_hand"]["x"], lm["r_hand"]["y"], lm["r_hand"]["z"]],
                    })
                name = slug(sign)
                with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"sign": sign, "fps": 15, "source": "LSCS45", "frames": frames}, f)
                index[name] = {"label": sign, "video": None, "source": "LSCS45"}
                print(name, len(frames), "frames")

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
