#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;
char s[N];

int main() {
    int T; scanf("%d", &T);
    while (T --) {
        int n, k; scanf("%d%d%s", &n, &k, s + 1);
        int ans = 0, last = 0, front = 0;
        for (int i = 1; i <= n; ++ i)
            if (s[i] == '1') {
                if (!front) front = i;
                last = i;
                ans += 11;
            }
        if (!front && !last) puts("0");
        else {
            int tofront = front - 1, totail = n - last;
            if (front == last) {
                if (totail <= k) ans -= 10;
                else if (tofront <= k) ans -= 1;
            }
            else {
                if (totail <= k) ans -= 10, k -= totail;
                if (tofront <= k) ans -= 1;
            }
            printf("%d\n", ans);
        }
    }
}