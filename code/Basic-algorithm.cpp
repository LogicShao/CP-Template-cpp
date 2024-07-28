#include <bits/stdc++.h>

using LL = long long;
using ULL = unsigned long long;
using LD = long double;
using PII = std::pair<int, int>;

// 快速幂
int fpow(int a, int b, int mod) {
    int ans = 1 % mod;
    for (; b; b >>= 1) {
        if (b & 1) ans = (LL)ans * a % mod;
        a = (LL)a * a % mod;
    }
    return ans;
}
// 龟速乘
int smul(int a, int b, int mod) {
    int ans = 0;
    for (; b; b >>= 1) {
        if (b & 1) ans = (ans + a) % mod;
        a = a * 2 % mod;
    }
    return ans;
}
// 快速乘
ULL qmul(ULL a, ULL b, ULL mod) {
    a %= mod, b %= mod;
    ULL c = (LD)a / mod * b;
    ULL x = a * b, y = c * mod;
    LL res = (LL)(x % mod) - (LL)(y % mod);
    return res < 0 ? res + mod : res;
}

// 排序
void quick_sort(int l, int r, int a[]) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = a[rand() % (r - l + 1) + l];
    while (i < j) {
        do i++; while (a[i] < x);
        do j--; while (a[j] > x);
        if (i < j) std::swap(a[i], a[j]);
    }
    quick_sort(l, j, a), quick_sort(j + 1, r, a);
}
// 快速选择
int quick_select(int l, int r, int k, int a[]) {
    if (l >= r) return a[l];
    int i = l - 1, j = r + 1, x = a[rand() % (r - l + 1) + l];
    while (i < j) {
        do i++; while (a[i] < x);
        do j--; while (a[j] > x);
        if (i < j) std::swap(a[i], a[j]);
    }
    if (j - l + 1 >= k) return quick_select(l, j, k, a);
    return quick_select(j + 1, r, k - (j - l + 1), a);
}
// 归并排序
void merge_sort(int l, int r, int a[], int b[]) {
    if (l >= r) return;
    int mid = (l + r) >> 1;
    merge_sort(l, mid, a, b), merge_sort(mid + 1, r, a, b);
    int i = l, j = mid + 1, k = l;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) b[k++] = a[i++];
        else b[k++] = a[j++];
    }
    while (i <= mid) b[k++] = a[i++];
    while (j <= r) b[k++] = a[j++];
    for (int i = l; i <= r; i++) a[i] = b[i];
}
// 归并排序求逆序对
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

// 高精度 由 AI 生成 不保证正确性
template <int Bits, int Base> // 表示位数和进制
struct HP {
    static const int LEN = Bits / log10(Base) + 1; // 计算数组长度
    int a[LEN]; // 数组 倒序存储
    bool neg; // 符号

    HP() : neg(false) { memset(a, 0, sizeof(a)); }

    HP(int x) : neg(x < 0) {
        memset(a, 0, sizeof(a));
        if (x < 0) x = -x;
        for (int i = 0; x; i++) {
            a[i] = x % Base;
            x /= Base;
        }
    }

    HP(const char *s) {
        memset(a, 0, sizeof(a));
        int len = strlen(s);
        if (s[0] == '-') neg = true, s++, len--;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len - i - 1; j++) {
                a[i] *= 10;
                if (a[i] >= Base) a[i + 1] += a[i] / Base, a[i] %= Base;
            }
            a[len - 1] += s[i] - '0';
            if (a[len - 1] >= Base) a[len] += a[len - 1] / Base, a[len - 1] %= Base;
        }
        if (a[len]) len++;
        while (len > 1 && !a[len - 1]) len--;
        neg = (len == 1 && a[0] == 0) ? false : neg;
        for (int i = 0; i < len; i++) a[i] = a[i] % Base;
        memset(a + len, 0, sizeof(a) - len);
    }
};