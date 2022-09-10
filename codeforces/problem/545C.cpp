#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10, INF = 2.1e9;
int x[N], h[N], dp[N][3];

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int n; cin >> n;
    for (int i = 1; i <= n; ++ i) cin >> x[i] >> h[i];
    x[n + 1] = INF, x[0] = -INF;

    for (int i = 1; i <= n; ++ i) {
        dp[i][0] = *max_element(dp[i - 1], dp[i - 1] + 3);
        if (x[i - 1] + h[i - 1] < x[i] - h[i]) {
            dp[i][1] = dp[i][0] + 1;
        }
        else if (x[i - 1] < x[i] - h[i]) {
            dp[i][1] = max(dp[i - 1][0], dp[i - 1][1]) + 1;
        }

        if (x[i] + h[i] < x[i + 1]) {
            dp[i][2] = dp[i][0] + 1;
        }
    }

    cout << *max_element(dp[n], dp[n] + 3) << '\n';
}