#include <iostream>
#include <cstring>
#include <algorithm>
#define toMax(a, b) a < b ? a = b: 0

using namespace std;

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n, maxval = -1e9, maxgrowth = -1e9; cin >> n;
        for (int i = 1; i <= n; ++ i) {
            int x; cin >> x;
            toMax(maxval, x);
            toMax(maxgrowth, maxval - x);
        }
        int res = 0;
        for (; maxgrowth > 0; maxgrowth >>= 1) ++ res;
        cout << res << '\n';
    }
}