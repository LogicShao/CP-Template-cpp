files = [
    "1_基础算法.md",
    "2_数据结构.md",
    "3_搜索与图论.md",
    "4_数学知识.md",
    "5_动态规划.md",
    "6_网络流初步.md",
    "7_图论II.md"
]
out = "README.md"


def filelines(file):
    res = []
    with open(file, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line != '' and not line.isspace():
                line = line.split()
                if len(line) > 1 and line[0].count("#") == len(line[0]):
                    res.append([len(line[0]), " ".join(line[1:])])
    return res


def makemenu(file):
    res = "## [%s](./%s)\n\n" % (file.replace(".md", "")[2:], file)
    s = filelines(file)
    det = min(map(lambda x: x[0], s))
    for dep, title in s:
        res += "    " * (dep - det) + "* " + title + "\n"
    return res + "\n"


if __name__ == "__main__":
    with open(out, "w", encoding="UTF-8") as f:
        f.write("""# Templates\n\n""")
        for i in files:
            f.write(makemenu(i))
