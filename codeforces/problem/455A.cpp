#include <bits/stdc++.h>
#define toMax(a, b) a < b ? a = b: 0

using namespace std;

typedef long long LL;

const int N = 1e5 + 10;
LL a[N], dp[N][2];

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int n, maxval = -1; cin >> n;
    while (n --) {
        int x; cin >> x;
        toMax(maxval, x);
        ++ a[x];
    }
    for (int i = 1; i <= maxval; ++ i) a[i] *= i;
    for (int i = 1; i <= maxval; ++ i) {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
        if (a[i - 1]) dp[i][1] = dp[i - 1][0] + a[i];
        else dp[i][1] = dp[i][0] + a[i];
    }
    cout << max(dp[maxval][0], dp[maxval][1]) << '\n';
}