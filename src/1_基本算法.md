# 基本算法

## 位运算

### 快速幂

```cpp
int fpow(int a, int b, int c) {
    int res = 1 % c;
    for (; b; b >>= 1) {
        if (b & 1) res = (long long)res * a % c;
        a = (long long)a * a % c;
    }
    return res;
}
```

### 龟速乘

```cpp
LL mul(LL a, LL b, LL c) {
    LL res = 0;
    for (; b; b >>= 1) {
        if (b & 1) res = (res + a) % c;
        a = (a + a) % c;
    }
    return res;
}
```

### 快速乘

```cpp
ull mul(ull a, ull b, ull p) {
    a %= p, b %= p;
    ull c = (long double)a * b / p;
    ull x = a * b, y = c * p;
    ll res = (ll)(x % p) - (ll)(y % p);
    if (res < 0) res += p;
    return res;
}
```

## 排序算法

### 快速排序

```c++
void q_sort(int l, int r) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = a[rand() % (r - l + 1) + l];
    while (i < j) {
        do i ++; while (x > a[i]);
        do j --; while (x < a[j]);
        if (i < j) swap(a[i], a[j]);
    }
    q_sort(l, j), q_sort(j + 1, r);
}
```

### 第 k 大数-快速选择算法

```cpp
void quick_select(std::vector<int> &a, int l, int r, int k) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = a[rand() % (r - l + 1) + l];
    while (i < j) {
        do ++i; while (a[i] > x);
        do --j; while (a[j] < x);
        if (i < j) std::swap(a[i], a[j]);
    }
    if (j - l + 1 >= k) quick_select(a, l, j, k);
    else quick_select(a, j + 1, r, k - (j - l + 1));
}
```

### 归并排序

```c++
void merge_sort(int l, int r) {
    if (l >= r) return;
    int mid = l + r >> 1;
    merge_sort(l, mid), merge_sort(mid + 1, r);
    int i = l, j = mid + 1, k = l;
    while (i <= mid && j <= r)
        if (a[i] < a[j]) b[k ++] = a[i ++];
        else b[k ++] = a[j ++];
    while (i <= mid) b[k ++] = a[i ++];
    while (j <= r) b[k ++] = a[j ++];
    for (int i = l; i <= r; i ++) a[i] = b[i];
}
```

### 归并排序求逆序对

```cpp
LL merge_sort(int l, int r, std::vector<int> &a) {
    if (l >= r) return 0;
    int mid = (l + r) >> 1;
    LL res = merge_sort(l, mid, a) + merge_sort(mid + 1, r, a);
    std::vector<int> b(r - l + 1);
    int i = l, j = mid + 1, k = l;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) b[k++] = a[i++];
        else b[k++] = a[j++], res += mid - i + 1;
    }
    while (i <= mid) b[k++] = a[i++];
    while (j <= r) b[k++] = a[j++];
    for (int i = l; i <= r; i++) a[i] = b[i];
    return res;
}
```

## 二分

### 整数二分算法模板

```c++
bool check(int x) {/* ... */} // 检查x是否满足某种性质
```

### 1. 区间 $[l, r]$ 被划分成 $[l,mid]$ 和 $[mid+1,r]$ 时使用

```c++
// 也就是说左边区间是答案
int bsearch_1(int l, int r) {
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
```

### 2. 区间 $[l,r]$ 被划分成 $[l,mid-1]$ 和 $[mid,r]$时使用

```c++
// 也就是说右边区间是答案
int bsearch_2(int l, int r) {
    while (l < r) {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

### 3. 浮点数二分算法模板

```c++
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r) {
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    // 建议使用固定次数的模板而不是控制精度的
    // 精度太高可能出现错误
    while (r - l > eps) {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```

## 高精度

### 1. 加法

```c++
vector<int> add(vector<int> &a, vector<int> &b) {
    vector<int> c;
    int t = 0; // 进位
    for (int i = 0; i < a.size() || i < b.size(); i++){
        if (i < a.size()) t += a[i];
        if (i < b.size()) t += b[i];
        c.push_back(t % 10);
        t /= 10;//进位权重下降
    }
    if (t) c.push_back(1);
    return c;
}
```

### 2. 减法

```c++
bool cmp(vector<int> &a, vector<int> &b) {
    if (a.size() != b.size()) return a.size() > b.size();
    for (int i = a.size(); i >= 0; i--)
        if (a[i] != b[i])
            return a[i] > b[i];
    return true;
}

vector<int> sub(vector<int> &a, vector<int> &b){
    vector<int> c;
    int t = 0; // 借位
    for (int i = 0; i < a.size(); i++){
        t = a[i] - t;
        if (i < b.size()) t -= b[i];
        c.push_back((t + 10) % 10);
        if (t < 0) t = 1; // t<0 表示借位了
        else t = 0; // 否则就是没借位
    }
    while (c.size() > 1 && c.back() == 0) c.pop_back();
    return c;
}
```

### 3. 高精度乘低精度

```c++
vector<int> mul(vector<int> &a, int b){
    vector<int> c;
    for (int i = 0, t = 0; i < a.size() || t; i++){//进位存在或没乘完
        if (i < a.size()) t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    while (c.size() > 1 && c.back() == 0) c.pop_back();//去除前导零
    return c;
}
```

### 4. 高精度乘高精度

```c++
vector<int> mul(vector<int> &a, vector<int> &b) {
    vector<int> c;
    c.resize(a.size() + b.size());
    for (int i = 0; i < a.size(); ++ i) {
        int t = 0;
        for (int j = 0; j < b.size(); ++ j) {
            c[i + j] += t + a[i] * b[j];
            t = c[i + j] / 10;
            c[i + j] %= 10;
        }
        c[i + b.size()] = t;
    }
    while (c.size() > 1 && c.back() == 0) c.pop_back();
    return c;
}
```

### 5. 高精度除低精度

```c++
vector<int> div(vector<int> &a, int b, int &r){
    vector<int> c;
    r = 0;
    for (int i = a.size() - 1; i >= 0; i--){
        r = r * 10 + a[i];
        c.push_back(r / b);
        r %= b;
    }
    reverse(c.begin(), c.end());
    while (c.size() > 1 && c.back() == 0) c.pop_back();//去除前导零
    return c;
}
```

## 前缀和差分

### 一维前缀和

```c++
int n, m;
scanf("%d%d", &n, &m);
for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
for (int i = 1; i <= n; i++) b[i] = a[i] + b[i - 1];
while (m --) {
    int l, r;
    scanf("%d%d", &l, &r);
    printf("%d\n", b[r] - b[l - 1]);
}
```

### 二维前缀和

```c++
scanf("%d%d%d", &n, &m, &q);
for (int i = 1; i <= n; i++)
    for (int j = 1; j <= m; j++)
        scanf("%d", &a[i][j]);
for (int i = 1; i <= n; i++)
    for (int j = 1; j <= m; j++)
        b[i][j] = b[i - 1][j] + b[i][j - 1] - b[i - 1][j - 1] + a[i][j];
while (q --) {
    scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
    printf("%d\n", b[x2][y2] - b[x1 - 1][y2] - b[x2][y1 - 1] + b[x1 - 1][y1 - 1]);
}
```

### 一维差分

```c++
void in(int l, int r, int c) {//插入
    b[l] += c, b[r + 1] -= c;
}
int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    for (int i = 1; i <= n; i++) b[i] = a[i] - a[i - 1]; //构造差分数组
    for (int i = 1; i <= m; i++) {
        scanf("%d%d%d", &l, &r, &c);
        in(l, r, c);
    }
    for (int i = 1; i <= n; i++) b[i] = b[i] + b[i - 1]; //还原数组
    for (int i = 1; i <= n; i++) printf("%d ", b[i]);
}   
```

### 二维差分

```c++
void in(int x1, int y1, int x2, int y2, int c) { //插入
    s[x1    ][y1    ] += c;
    s[x2 + 1][y1    ] -= c;
    s[x1    ][y2 + 1] -= c;
    s[x2 + 1][y2 + 1] += c;
}
int main(){
    scanf("%d%d%d", &n, &m, &q);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            scanf("%d", &a[i][j]);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            in(i, j, i, j, a[i][j]);//构造
    for (int i = 1; i <= q; i++){
        scanf("%d%d%d%d%d", &x1, &y1, &x2, &y2, &c);
        in(x1, y1, x2, y2, c);
    }
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];//还原
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++){
            printf("%d ", s[i][j]);
            if (j == m) puts("");
        }
}
```

## 双指针算法

```c++
for (int i = 0, j = 0; i < n; i ++ ) {
    while (j < i && check(i, j)) j ++ ;
    // 具体问题的逻辑
}
```

常见问题分类：
* `(1)` 对于一个序列，用两个指针维护一段区间
* `(2)` 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

## 离散化+树状数组求逆序对

```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>

using namespace std;

typedef long long LL;

const int N = 5e5 + 10;
int c[N], n;
int a[N], b[N];
unordered_map<int, int> m;

int lowbit(int x) {
    return x & -x;
}

void add(int x, int v) {
    for (int i = x; i < N; i += lowbit(i))
        c[i] += v;
}

int query(int x) {
    int res = 0;
    for (int i = x; i; i -= lowbit(i))
        res += c[i];
    return res;
}

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; ++ i) 
        scanf("%d", &a[i]), b[i] = a[i];
    sort(b + 1, b + 1 + n);
    int cnt = 0;
    for (int i = 1, j = 1; i <= n; ++ i) {
        while (j <= n && b[j] == b[i]) ++ j;
        m[b[i]] = ++ cnt;
        i = j - 1;
    }
    for (int i = 1; i <= n; ++ i)
        a[i] = m[a[i]];
    LL ans = 0;
    for (int i = 1; i <= n; ++ i) {
        add(a[i], 1);
        ans += i - query(a[i]);
    }
    cout << ans << endl;
}
```

### 二分实现

```cpp
sort(b + 1, b + 1 + n);
int k = unique(b + 1, b + 1 + n) - b - 1;
for (int i = 1; i <= n; ++ i)
    a[i] = lower_bound(b + 1, b + 1 + k, a[i]) - b;
```

## 模拟退火

使用范围: 最优化问题，比如DP，贪心，计算几何

如果函数连续性较强 (即轻微的扰动对函数值的影响较小)，退火的出解率较高

序列的邻项考虑随机交换，如 [2424. 保龄球 - AcWing题库](https://www.acwing.com/problem/content/2426/)

```cpp
#include <bits/stdc++.h> // https://www.acwing.com/problem/content/3170/

using namespace std;

#define x first
#define y second
#define toMin(a, b) a > b ? a = b: 0

typedef pair<double, double> PDD;

const int N = 110;
double ans = 1e8;
PDD a[N];
int n;

double rand(int l, int r) {
    return (double)rand() / RAND_MAX * (r - l) + l;
}

double dist(PDD a, PDD b) {
    auto x = a.x - b.x;
    auto y = a.y - b.y;
    return sqrt(x * x + y * y);
}

double calc(PDD p) {
    double res = 0;
    for (int i = 1; i <= n; ++ i) res += dist(p, a[i]);
    toMin(ans, res);
    return res;
}

void SA() {
    PDD cur(rand(0, 10000), rand(0, 10000));
    for (double T = 1e4; T > 1e-4; T *= 0.99) { // 不同的参数出解率不同
        PDD np(rand(cur.x - T, cur.x + T), rand(cur.y - T, cur.y + T));
        double dt = calc(np) - calc(cur);
        if (exp(-dt / T) > rand(0, 1)) cur = np;
    }
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    srand(time(0));
    
    cin >> n;
    for (int i = 1; i <= n; ++ i) cin >> a[i].x >> a[i].y;
    for (int i = 1; i <= 100; ++ i) SA(); // 考场里可以考虑卡时
    cout << int(ans + 0.5) << '\n';
}
```

## 文件读写

***

```c++
// #include <cstdlib>
// freopen("P2058_2.in", "r", stdin);
// freopen("my_ans.out", "w", stdout);
// fclose(stdin);
// fclose(stdout);
```

## 快读

`cin` 加速：根据个人评测经验，这种方法会比 `scanf()` 还要快

```cpp
ios::sync_with_stdio(false);
cin.tie(nullptr);
cout.tie(nullptr);
```

使用  `getchar()`

```cpp
template<typename T> void read(T &x) {
    char ch = getchar(); bool flag = 0; x = 0;
    for (; ch < '0' || ch > '9'; ch = getchar())
        flag |= (ch == '-');
    for (; ch >= '0' && ch <= '9'; ch = getchar())
        x = (x << 1) + (x << 3) + ch - '0';
    if (flag) x = -x;
}
```

将字符先读到 `buff` 数组中再从 `buff` 中读入

```cpp
inline char GET_CHAR() {
    static char buf[maxn], *p1 = buf, *p2 = buf;
    return p1 == p2 && (p2 = (p1 = buf) + 
        fread(buf, 1, maxn, stdin), p1 == p2) ? EOF: *p1 ++;
}

template<class T> inline void read(T &x) {
    x = 0; int f = 0; char ch = GET_CHAR();
    for (; ch < '0' || ch > '9'; ch = GET_CHAR()) flag |= (ch == '-');
    for (; ch >= '0' && ch <= '9'; ch = GET_CHAR()) x = (x << 1) + (x << 3) + (ch ^ 48);
    x = f ? -x: x;
}
```
