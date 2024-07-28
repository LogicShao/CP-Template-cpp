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