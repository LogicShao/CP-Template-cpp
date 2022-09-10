#include <iostream>
#include <cstring>
#include <algorithm>
#define toMax(a, b) a < b ? a = b: 0

using namespace std;

typedef pair<int, int> PII;
typedef long long LL;

const int N = 1e5 + 10;
PII a[N];
LL b[N];

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int n, d; cin >> n >> d;
    for (int i = 1; i <= n; ++ i)
        cin >> a[i].first >> a[i].second;
    sort(a + 1, a + 1 + n);
    for (int i = 1; i <= n; ++ i)
        b[i] = b[i - 1] + a[i].second;
    LL res = 0;
    for (int i = 1; i <= n; ++ i) {
        int l = upper_bound(a + 1, a + 1 + i, PII{a[i].first - d + 1, 0}) - a;
        toMax(res, b[i] - b[l - 1]);
    }
    cout << res << endl;
}