files = [
    '1_基本算法.md', '2_数据结构.md', '3_搜索.md',
    '4_动态规划.md', '5_图论.md', '6_数学知识.md',
]
out = "menu.md"


def filelines(file):
    # return the file's titles
    res = []
    with open(file, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line != '' and not line.isspace():
                line = line.split()
                if len(line) > 1 and line[0].count("#") == len(line[0]):
                    res.append([len(line[0]), " ".join(line[1:])])
    return res


def removespace(file):
    # remove spaces in file's titles
    s = ''
    with open(file, "r", encoding="UTF-8") as f:
        for i in f.readlines():
            flag = False

            line = i.strip()
            if line != '' and not line.isspace():
                line = line.split()
                if len(line) > 1 and line[0].count("#") == len(line[0]):
                    flag = True

            if flag:
                s += line[0] + " " + "".join(line[1:]) + "\n"
            else:
                s += i

    with open(file, "w", encoding="UTF-8") as f:
        f.write(s)


def makemenu(file):
    res = "## [%s](./%s)\n\n" % (file.replace(".md", "")[2:], file)
    s = filelines(file)
    det = min(map(lambda x: x[0], s))
    for dep, title in s:
        tmp = "    " * (dep - det) + "* "
        tmp += "[%s](./%s#%s)\n" % (title, file, title)
        res += tmp
    return res + "\n"


if __name__ == "__main__":
    with open(out, "w", encoding="UTF-8") as f:
        for i in files:
            # removespace(i)
            f.write(makemenu(i))
