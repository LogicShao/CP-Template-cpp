import os


folder = '.\\src\\'
output = '.\\'
filename = 'allinone.md'


def getMdFiles(folder):
    files = os.listdir(folder)
    mdFiles = []
    for file in files:
        if file.endswith('.md'):
            mdFiles.append(file)
    return mdFiles


def getMd(file):
    with open(file, "r", encoding="UTF-8") as f:
        return f.readlines()


def memgeMd(mdFiles, folder):
    res = []
    for file in mdFiles:
        res += getMd(folder + file)
    return res


if __name__ == '__main__':
    mdFiles = getMdFiles(folder)
    res = memgeMd(mdFiles, folder=folder)
    with open(output + filename, "w", encoding="UTF-8") as f:
        for line in res:
            f.write(line)