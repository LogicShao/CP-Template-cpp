#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#define debug
using namespace std;

typedef pair<int, int> PII;

const int N = 1e5 + 10;
bool color[N], lzdel[N];
int f[N];

struct seg {
    int pos, id;
    bool is_l;
    bool operator< (const seg &x) const {
        return pos < x.pos;
    }
    void out(string info = "") {
printf("id:%d color:%d pos:%d %c %s\n", id, color[id], pos, "rl"[is_l], info.c_str());
    }
    string info(string add = "") {
        string res ="seg{"+to_string(id)+","+to_string(pos)+","+(is_l?"l":"r")+"}"+add;
        return res;
    }
} a[N];

struct queue {
    int hh = 0, tt = -1, q[N];
    void clear() { hh = 0, tt = -1; }
    void push(int x) { q[++ tt] = x; }
    int front() { return q[hh]; }
    void pop() { ++ hh; }
    bool empty() { return tt < hh; }
    int size() { return tt - hh + 1; }
    void out(string info = "") {
        cout << "--q: [";
        for (int i = hh; i <= tt; ++ i) cout << q[i] << ' ';
        cout << "]-- " << info << endl;
    }
} q[2];

int find(int x) {
    return x == f[x] ? x: f[x] = find(f[x]);
}

void merge(int x, int y) {
    f[find(x)] = find(y);
}

int main() {
#ifdef debug
freopen("E.in", "r", stdin);
freopen("E.out", "w", stdout);
#endif

    int T; scanf("%d", &T);
    while (T --) {
        int n; scanf("%d", &n);
        for (int i = 1; i <= n; ++ i) {
            int c, l, r;
            scanf("%d%d%d", &c, &l, &r);
            a[i * 2 - 1] = { l, i, true };
            a[i * 2] = { r, i, false };
            color[i] = c;
            f[i] = i;
        }

        sort(a + 1, a + 1 + n * 2);
        q[0].clear(), q[1].clear();
        memset(lzdel, false, sizeof lzdel);

#ifdef debug
cout << "---------------------" << endl;
for (int i = 1; i <= n * 2; ++ i) a[i].out(to_string(i));
cout << "---------------------" << endl;
#endif

        for (int i = 1; i <= n * 2; ++ i) {
            int id = a[i].id, c = color[id];
            if (a[i].is_l) {
                q[c].push(i);
#ifdef debug
printf("push %s to q[%d]\n", a[i].info(to_string(i)).c_str(), c);
#endif
                c ^= 1;
                while (q[c].size() > 1) {
                    int j = q[c].front();
                    q[c].pop();
                    if (!lzdel[a[j].id]) merge(id, a[j].id);
#ifdef debug
string mge = !lzdel[a[j].id] ? (to_string(id)+" "+to_string(a[j].id)): "NO";
cout << "---------------------" << endl;
printf("pop %s merge:%s\n", a[j].info(to_string(j)).c_str(), mge.c_str());
a[j].out();
q[c].out(to_string(c));
#endif
                }
                if (!q[c].empty()) {
                    int j = q[c].front();
                    if (!lzdel[a[j].id]) merge(id, a[j].id);
#ifdef debug
string mge = !lzdel[a[j].id] ? (to_string(id)+" "+to_string(a[j].id)): "NO";
cout << "---------------------" << endl;
printf("pop %s merge:%s\n", a[j].info(to_string(j)).c_str(), mge.c_str());
a[j].out();
q[c].out(to_string(c));
#endif
                }
            } else {
                lzdel[id] = true;
#ifdef debug
cout << "lazy del " << id << endl;
#endif
            }
        }

        int res = 0;
        for (int i = 1; i <= n; ++ i) res += (i == f[i]);

        printf("%d\n", res);
    }
}