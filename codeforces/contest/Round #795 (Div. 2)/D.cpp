#include <iostream>
#include <cstring>
#include <algorithm>

#define debug

using namespace std;

typedef long long LL;

const int N = 2e5 + 10;
const LL INF = __LONG_LONG_MAX__;
LL a[N], b[N];
int nxt_up[N], pre_up[N];

struct stack {
    int tt, a[N];
    void clear() { tt = 0; }
    void push(int x) { a[++ tt] = x; }
    void pop() { -- tt; }
    int top() { return a[tt]; }
    int size() { return tt; }
} stk;

#define ls(x) x << 1
#define rs(x) x << 1 | 1

struct node {
    int l, r;
    LL maxv, minv;
} tr[N << 2];

void pushup(int u) {
    tr[u].maxv = max(tr[ls(u)].maxv, tr[rs(u)].maxv);
    tr[u].minv = min(tr[ls(u)].minv, tr[rs(u)].minv);
}

void build(int u, int l, int r) {
    if (l == r) tr[u] = { l, r, b[l], b[l] };
    else {
        tr[u] = { l, r };
        int mid = l + r >> 1;
        build(ls(u), l, mid), build(rs(u), mid + 1, r);
        pushup(u);
    }
}

LL querymax(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].maxv;
    int mid = tr[u].l + tr[u].r >> 1;
    LL res = -INF;
    if (l <= mid) res = max(res, querymax(ls(u), l, r));
    if (r > mid) res = max(res, querymax(rs(u), l, r));
    return res;
}

LL querymin(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].minv;
    int mid = tr[u].l + tr[u].r >> 1;
    LL res = INF;
    if (l <= mid) res = min(res, querymin(ls(u), l, r));
    if (r > mid) res = min(res, querymin(rs(u), l, r));
    return res;
}

int main() {
    int T; scanf("%d", &T);
    while (T --) {
        int n; scanf("%d", &n);
        for (int i = 1; i <= n; ++ i) {
            scanf("%lld", a + i);
            b[i] = b[i - 1] + a[i];
        }

        build(1, 0, n);

        stk.clear();
        for (int i = 1; i <= n; ++ i) {
            while (stk.size() && a[stk.top()] <= a[i]) stk.pop();
            pre_up[i] = stk.size() ? stk.top(): 0;
            stk.push(i);
        }
        stk.clear();
        for (int i = n; i; -- i) {
            while (stk.size() && a[stk.top()] <= a[i]) stk.pop();
            nxt_up[i] = stk.size() ? stk.top(): n + 1;
            stk.push(i);
        }

        bool flag = true;
        for (int i = 1; i <= n && flag; ++ i) {
            int l = pre_up[i], r = nxt_up[i];
            LL m = querymax(1, i, r - 1) - querymin(1, l, i - 1);
            if (a[i] < m) flag = false;
        }

        puts(flag ? "YES": "NO");
    }
}