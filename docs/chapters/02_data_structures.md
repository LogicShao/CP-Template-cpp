# 数据结构

## 链表

### 1单链表

```c++
#include <iostream>
using namespace std;
const int N = 100010;
int head, e[N], ne[N], idx;
void init() {
    head = -1;
    idx = 0;
}
void add_to_head(int x) {
    e[idx] = x, ne[idx] = head, head = idx ++ ;
}
void add(int k, int x) {
    e[idx] = x, ne[idx] = ne[k], ne[k] = idx ++ ;
}
void remove(int k) {
    ne[k] = ne[ne[k]];
}
int main() {
    int m; cin >> m;
    init();
    while (m -- ) {
        int k, x; char op;
        cin >> op;
        if (op == 'H') {
            cin >> x;
            add_to_head(x);
        } else if (op == 'D') {
            cin >> k;
            if (!k) head = ne[head];
            else remove(k - 1);
        } else {
            cin >> k >> x;
            add(k - 1, x);
        }
    }
    for (int i = head; i != -1; i = ne[i]) cout << e[i] << ' ';
    cout << endl;
    return 0;
}
```

### 2双链表

```c++
#include <iostream>
using namespace std;
const int N = 100010;
int m;
int e[N], l[N], r[N], idx;
// 在节点a的右边插入一个数x
void insert(int a, int x) {
    e[idx] = x;
    l[idx] = a, r[idx] = r[a];
    l[r[a]] = idx, r[a] = idx ++ ;
}
// 删除节点a
void remove(int a) {
    l[r[a]] = l[a];
    r[l[a]] = r[a];
}
int main() {
    cin >> m;
    // 0是左端点，1是右端点
    r[0] = 1, l[1] = 0;
    idx = 2;
    while (m -- ) {
        string op; cin >> op;
        int k, x;
        if (op == "L"){
            cin >> x;
            insert(0, x);
        } else if (op == "R"){
            cin >> x;
            insert(l[1], x);
        } else if (op == "D") {
            cin >> k;
            remove(k + 1);
        } else if (op == "IL") {
            cin >> k >> x;
            insert(l[k + 1], x);
        } else {
            cin >> k >> x;
            insert(k + 1, x);
        }
    }
    for (int i = r[0]; i != 1; i = r[i]) cout << e[i] << ' ';
    cout << endl;
    return 0;
}
```



## 栈

```c++
#include <iostream>
#include <cstring>
using namespace std;
const int N = 1e6 + 10;
int stk[N], tt = -1;
string op;
int m, k;
int main() {
    cin >> m;
    while (m --) {
        cin >> op;
        if (op == "push") {
            cin >> k;
            stk[++ tt] = k;
        } else if (op == "query") {
            cout << stk[tt] << endl;
        } else if (op == "pop") tt --;
        else cout << (tt < 0 ? "YES" : "NO") << endl;
    }
    return 0;
}
```

### 单调栈

```c++
#include <iostream>
#include <cstdio>

using namespace std;

const int N = 3e6 + 10;
int stk[N], tt, n;
int a[N], ans[N];

void push(int x) { stk[++tt] = x; }

int pop() { return stk[tt--]; }

int top() { return stk[tt]; }

int empty()
{
    if (tt > 0)
        return false;
    return true;
}

void out()
{
    for (int i = 1; i <= tt; i++)
        cout << stk[i] << ' ';
    cout << endl;
}

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        scanf("%d", &a[i]);
    for (int i = n; i > 0; i--)
    {
        while (!empty() && a[top()] <= a[i]) pop();
        ans[i] = empty() ? 0 : top();
        push(i);
    }
    for (int i = 1; i <= n; i++) printf("%d ", ans[i]);
    //system("pause");
    return 0;
}
```

## 队列

```c++
#include <iostream>
using namespace std;
const int N = 100010;
int m;
int q[N], hh, tt = -1;
int main() {
    cin >> m;
    while (m -- ) {
        string op;
        int x;
        cin >> op;
        if (op == "push") {
            cin >> x;
            q[ ++ tt] = x;
        }
        else if (op == "pop") hh ++ ;
        else if (op == "empty") cout << (hh <= tt ? "NO" : "YES") << endl;
        else cout << q[hh] << endl;
    }
    return 0;
}
```

### 单调队列

```c++
#include <iostream>
#include <cstdio>

using namespace std;

const int N = 1e6 + 10;
int q[N], hh, tt = -1;
int n, k;
int a[N];

int main()
{
    scanf("%d%d", &n, &k);
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &a[i]);
        if (i - k + 1 > q[hh])
            ++hh;
        while (hh <= tt && a[i] <= a[q[tt]])
            --tt;
        q[++tt] = i;
        if (i + 1 >= k)
            printf("%d ", a[q[hh]]);
    }
    puts("");
    hh = 0, tt = -1;
    for (int i = 0; i < n; i++)
    {
        if (i - k + 1 > q[hh])
            ++hh;
        while (hh <= tt && a[i] >= a[q[tt]])
            --tt;
        q[++tt] = i;
        if (i + 1 >= k)
            printf("%d ", a[q[hh]]);
    }
    //system("pause");
    return 0;
}
```

## KMP

```c++
// KMP
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 1e6 + 10;
int n, m;
char p[N] = "0", s[N] = "0";
int ne[N];

int main()
{
    cin >> s + 1 >> p + 1;
    n = strlen(p) - 1;
    m = strlen(s) - 1;

    for (int i = 2, j = 0; i <= n; i++)
    {
        while (j && p[i] != p[j + 1])
            j = ne[j];
        if (p[i] == p[j + 1])
            j++;
        ne[i] = j;
    }

    for (int i = 1, j = 0; i <= m; i++)
    {
        while (j && s[i] != p[j + 1])
            j = ne[j];
        if (s[i] == p[j + 1])
            j++;
        if (j == n)
        {
            printf("%d\n", i - n + 1);
            j = ne[j];
        }
    }

    for (int i = 1; i <= n; i++)
        printf("%d ", ne[i]);
    system("pause");
    return 0;
}
```

## Trie树

```c++
const int M = 1e6 + 10;
int son[M][26], cnt[M], idx;
// 插入
void insert(char str[])
{
    int p = 0;
    for (int i = 0; str[i]; i++)
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];  
    }
    cnt[p] ++;
}
// 查询
int query(char str[])
{
    int p = 0;
    for (int i = 0; str[i]; i++)
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}
```

## 并查集

```c++
void init(int n) {
    for (int i = 1; i <= n; i ++) fa[i] = i;
}

int find(int x) {
    return x == fa[x] ? x : fa[x] = find(fa[x]);
}

void merge(int x, int y) {
    fa[find(x)] = find(y);
}
```

## 堆

```c++
#include <iostream>
#include <cstdio>

using namespace std;

const int N = 1e6 + 10;
int h[N], cnt, n;

void up(int k) {
    while (k >> 1 && h[k >> 1] > h[k]) {
        swap(h[k], h[k >> 1]);
        k >>= 1;
    }
}

void insert(int x) {
    h[++ cnt] = x;
    up(cnt);
}

void down(int x) {
    int t = x;
    if (x * 2 <= cnt && h[t] > h[x * 2]) t = 2 * x;
    if (2 * x + 1 <= cnt && h[t] > h[2 * x + 1]) t = 2 * x + 1;
    if (t != x) swap(h[t], h[x]), down(t);
}

void remove() {
    swap(h[1], h[cnt--]);
    down(1);
}

int main() {
    scanf("%d", &n);
    int op, x;
    while (n --) {
        scanf("%d", &op);
        if (op == 1) {
            scanf("%d", &x);
            insert(x);
        }
        else if (op == 2) printf("%d\n", h[1]);
        else if (op == 3) remove();
    }
    return 0;
}
```

## hash表


### 1开放寻址法

```c++
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;

const int N = 2e5 + 10, INF = 0x3f3f3f3f;
int h[N];
int find(int x) {
    int t = (x % N + N) % N;
    while (h[t] != INF && h[t] != x) {
        t ++;
        if (t == N) t = 0;
    }
    return t;
}

int main() {
    int n;
    scanf("%d", &n);
    memset(h, 0x3f, sizeof h);
    while (n --) {
        char op[2]; int x;
        scanf("%s%d", op, &x);
        int k = find(x);
        if (op[0] == 'I') h[k] = x;
        else {
            if (h[k] == INF) puts("No");
            else puts("Yes");
        }
    }
}
```



### 2拉链法

```c++
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 1e5 + 3;

int h[N], e[N], ne[N], idx;

void insert(int x) {
    int k = (x % N + N) % N;
    e[idx] = x, ne[idx] = h[k], h[k] = idx ++;
}

bool find(int x) {
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i]) if (e[i] == x) return true;
    return false;
}

int main() {
    int n;
    scanf("%d", &n);
    memset(h, -1, sizeof h);
    while (n --) {
        char op[2]; int x;
        scanf("%s%d", op, &x);
        if (op[0] == 'I') insert(x);
        else {
            if (find(x)) puts("Yes");
            else puts("No");
        }
    }
}
```

### 3字符串哈希

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10, P = 131;
#define u unsigned long long 
u n, m, h[N], p[N];
char str[N];
u f(int l, int r) {
    return h[r] - h[l - 1] * p[r - l + 1];
}
int main() {
    cin >> n >> m >> str + 1; p[0] = 1;
    for(int i = 1; i <= n; i ++) {
        h[i] = h[i - 1] * P + str[i];
        p[i] = p[i - 1] * P;
    }
    while(m --) {
        u a, b,c, d;
        cin >> a >> b >> c >> d;
        if(f(a, b) == f(c, d)) puts("Yes");
        else puts("No");
    }
    return 0;
}
```

随机模数，防止 hack。

```c++
using LL = long long;
using i28 = __int128; // gcc win 64 only

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

LL fpow(LL a, LL b, LL p) {
    LL res = 1;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return res;
}

bool millerRabin(LL x) {
    if (x < 2) return false;
    if (x == 2) return true;
    if (x % 2 == 0) return false;
    LL p = x - 1, q = 0;
    while (!(p & 1)) p >>= 1, q++;
    for (int i = 0; i < 10; i++) {
        LL a = rng() % (x - 2) + 2;
        LL v = fpow(a, p, x);
        if (v == 1 || v == x - 1) continue;
        for (int j = 0; j < q; j++) {
            v = v * v % x;
            if (v == x - 1) break;
            if (j == q - 1) return false;
        }
    }
    return true;
}

template<typename T1 = int, typename T2 = LL> // T1用来存储数据 T2用来进行预算(防止溢出)
struct strHash {
    int base, n;
    std::vector<T1> hash, pow;
    T1 mod;

    strHash(int N, int base = 131) : base(base), pow(N) {
        mod = std::uniform_int_distribution<T1>(1e9, 2e9)(rng);
        while (!millerRabin(mod)) mod++;
        pow[0] = 1;
        for (int i = 1; i < N; i++) pow[i] = (T2)pow[i - 1] * base % mod;
    }

    void init(std::string &s, int n) { // s下标从1开始
        this->n = n;
        hash.assign(n + 1, 0);
        for (int i = 1; i <= n; i++) hash[i] = ((T2)hash[i - 1] * base + s[i]) % mod;
    }

    T1 get(int l, int r) { // [l, r]
        return (hash[r] - (T2)hash[l - 1] * pow[r - l + 1] % mod + mod) % mod;
    }
};
```

## C++STL

```c++
unique, 将数组中重复的元素放到了最后
sort, 将数组中的元素排序
vector, 变长数组，倍增的思想
    size()  返回元素个数
    empty()  返回是否为空
    clear()  清空
    front()/back()
    push_back()/pop_back()
    begin()/end()
    []
    支持比较运算，按字典序
pair<int, int>
    first, 第一个元素
    second, 第二个元素
    支持比较运算，以first为第一关键字，以second为第二关键字（字典序）
string，字符串
    size()/length()  返回字符串长度
    empty()
    clear()
    substr(起始下标，(子串长度))  返回子串
    c_str()  返回字符串所在字符数组的起始地址
queue, 队列
    size()
    empty()
    push()  向队尾插入一个元素
    front()  返回队头元素
    back()  返回队尾元素
    pop()  弹出队头元素
priority_queue, 优先队列，默认是大根堆
    push()  插入一个元素
    top()  返回堆顶元素
    pop()  弹出堆顶元素
    定义成小根堆的方式：priority_queue<int, vector<int>, greater<int>> q;
stack, 栈
    size()
    empty()
    push()  向栈顶插入一个元素
    top()  返回栈顶元素
    pop()  弹出栈顶元素
deque, 双端队列
    size()
    empty()
    clear()
    front()/back()
    push_back()/pop_back()
    push_front()/pop_front()
    begin()/end()
    []
set, map, multiset, multimap, 基于平衡二叉树（红黑树），动态维护有序序列
    size()
    empty()
    clear()
    begin()/end()
    ++, -- 返回前驱和后继，时间复杂度 O(logn)

    set/multiset
        insert()  插入一个数
        find()  查找一个数
        count()  返回某一个数的个数
        erase()
            (1) 输入是一个数x，删除所有x   O(k + logn)
            (2) 输入一个迭代器，删除这个迭代器
        lower_bound()/upper_bound()
            lower_bound(x)  返回大于等于x的最小的数的迭代器
            upper_bound(x)  返回大于x的最小的数的迭代器
    map/multimap
        insert()  插入的数是一个pair
        erase()  输入的参数是pair或者迭代器
        find()
        []  注意multimap不支持此操作。 时间复杂度是 O(logn)
        lower_bound()/upper_bound()
unordered_set, unordered_map, unordered_multiset, unordered_multimap, 哈希表
    和上面类似，增删改查的时间复杂度是 O(1)
    不支持 lower_bound()/upper_bound()， 迭代器的++，--
bitset, 圧位
    bitset<10000> s;
    ~, &, |, ^
    >>, <<
    ==, !=
    []

    count()  返回有多少个1

    any()  判断是否至少有一个1
    none()  判断是否全为0

    set()  把所有位置成1
    set(k, v)  将第k位变成v
    reset()  把所有位变成0
    flip()  等价于~
    flip(k) 把第k位取反
```

## C++pb_ds

洛谷 P3369 【模板】普通平衡树

```c++
#include <iostream>
#include <ext/pb_ds/assoc_container.hpp> // 引入树的头文件
#include <ext/pb_ds/tree_policy.hpp>

/**
 * 赛时可以写万能头
 * #include <bits/stdc++.h>
 * #include <bits/extc++.h>
 */

namespace pbds = __gnu_pbds;

template<typename T, typename cmp = std::less<T>>
using rbtree = pbds::tree<T, pbds::null_type, cmp, pbds::rb_tree_tag, pbds::tree_order_statistics_node_update>;
using PII = std::pair<int, int>;

void solve() {
    int n;
    std::cin >> n;

    rbtree<PII> tree;
    int idx = 0;

    while (n --) {
        int op, x;
        std::cin >> op >> x;
        
        if (op == 1) {
            tree.insert({x, ++ idx});
        } else if (op == 2) {
            tree.erase(tree.lower_bound({x, 0}));
        } else if (op == 3) {
            std::cout << tree.order_of_key({x, 0}) + 1 << '\n';
        } else if (op == 4) {
            std::cout << tree.find_by_order(x - 1)->first << '\n';
        } else if (op == 5) {
            std::cout << (--tree.lower_bound({x, 0}))->first << '\n';
        } else {
            std::cout << tree.upper_bound({x, idx})->first << '\n';
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    solve();
}
```


## 线段树

```c++
#include <bits/stdc++.h>

using LL = long long;

struct node {
    int l, r;
    LL val, add;
};

struct segmentTree {
    #define ls(x) (x << 1)
    #define rs(x) (x << 1 | 1)
    #define p tr[u]
    #define pl tr[ls(u)]
    #define pr tr[rs(u)]

    std::vector<node> tr;
    segmentTree(int n) : tr(n << 2) {}

    void pushup(int u) {
        p.val = pl.val + pr.val;
    }

    void pushdown(int u) {
        if (p.add) {
            pl.val += p.add * (pl.r - pl.l + 1);
            pr.val += p.add * (pr.r - pr.l + 1);
            pl.add += p.add;
            pr.add += p.add;
            p.add = 0;
        }
    }

    void build(int u, int l, int r, std::vector<LL> &a) {
        p.l = l, p.r = r;
        if (l == r) {
            p.val = a[l];
            return;
        }
        int mid = (l + r) >> 1;
        build(ls(u), l, mid, a);
        build(rs(u), mid + 1, r, a);
        pushup(u);
    }

    void modify(int u, int l, int r, LL d) {
        if (p.l >= l && p.r <= r) {
            p.val += d * (p.r - p.l + 1);
            p.add += d;
            return;
        }
        pushdown(u);
        int mid = (p.l + p.r) >> 1;
        if (l <= mid) modify(ls(u), l, r, d);
        if (r > mid) modify(rs(u), l, r, d);
        pushup(u);
    }

    LL query(int u, int l, int r) {
        if (p.l >= l && p.r <= r) return p.val;
        pushdown(u);
        int mid = (p.l + p.r) >> 1;
        LL res = 0;
        if (l <= mid) res += query(ls(u), l, r);
        if (r > mid) res += query(rs(u), l, r);
        return res;
    }

    #undef ls
    #undef rs
    #undef p
    #undef pl
    #undef pr
};

void solve() {
    int n, m;
    std::cin >> n >> m;
    std::vector<LL> a(n + 1);
    for (int i = 1; i <= n; ++ i) std::cin >> a[i];
    segmentTree tree(n + 10);
    tree.build(1, 1, n, a);
    while (m --) {
        int op, x, y, k;
        std::cin >> op >> x >> y;
        if (op == 1) {
            std::cin >> k;
            tree.modify(1, x, y, k);
        } else {
            std::cout << tree.query(1, x, y) << '\n';
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    solve();
}
```

## 树状数组

```cpp
#include <bits/stdc++.h>

using namespace std;

const int N = 5e5 + 10;
int n, m, tr[N];

int lowbit(int x) {
    return x & -x;
}

void add(int x, int v) {
    for (; x < N; x += lowbit(x)) tr[x] += v;
}

int query(int x) {
    int res = 0;
    for (; x; x -= lowbit(x)) res += tr[x];
    return res;
}

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) {
        int t; scanf("%d", &t);
        add(i, t);
    }
    while (m --) {
        int op, x, y;
        scanf("%d%d%d", &op, &x, &y);
        if (op == 1) add(x, y);
        else printf("%d\n", query(y) - query(x - 1));
    }
}
```

## 回文串

马拉车 d1: 奇数长度最长回文半径 d2: 偶数长度最长回文半径

```cpp
void calc(string &s, int n, bool *pre) { // 判断前缀是否回文 字符串下标从0开始
    vector<int> d1(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
            k++;
        }
        d1[i] = k--;
        if (i + k > r) {
            l = i - k;
            r = i + k;
        }
    }
    vector<int> d2(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
        while (0 <= i - k - 1 && i + k < n && s[i - k - 1] == s[i + k]) {
            k++;
        }
        d2[i] = k--;
        if (i + k > r) {
            l = i - k - 1;
            r = i + k;
        }
    }
    for (int i = 0; i < n; ++ i) {
        int l = i - d1[i] + 1, r = i + d1[i] - 1;
        if (!l && !pre[r]) pre[r] = true;
 
        l = i - d2[i], r = i + d2[i] - 1;
        if (!l && !pre[r]) pre[r] = true;
    }
}
```
