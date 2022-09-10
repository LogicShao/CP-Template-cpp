#include <bits/stdc++.h>

using namespace std;

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n; cin >> n;
        cout << n << " \n"[n == 1];
        for (int i = 1; i < n; ++ i)
            cout << i << " \n"[i == n - 1];
    }
}