files = [
    "1_基本算法.md",
    "2_数据结构.md",
    "3_图论I.md",
    "4_数学知识.md",
    "5_动态规划.md",
    "6_网络流初步.md",
    "7_图论II.md",
    "8_搜索.md"
]
out = "README.md"
READMEinfo = """# Templates\n
这个仓库总结了部分算法竞赛模板  
但困于个人能力限制，本仓库只有提高组级别的模板  
而且个别模板质量并不高，也希望各位多多包涵  
如果你有更好的模板或者想法，我很乐意合并你的请求  

另外，本仓库中大部分模板取自 [AcWing算法全家桶](https://www.acwing.com/activity/content/2142/) 和 [算法进阶指南](https://www.acwing.com/activity/content/6/)  
知识结构也基本类似，而少部分来自网络或个人代码，感谢Ta们对我编程学习的帮助  

以下是目录:
"""


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
        f.write(READMEinfo)
        for i in files:
            # removespace(i)
            f.write(makemenu(i))
