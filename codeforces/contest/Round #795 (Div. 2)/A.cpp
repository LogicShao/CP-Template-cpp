#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

int main() {
    int T;
    scanf("%d", &T);
    while (T --) {
        int n, odd = 0, even = 0;
        scanf("%d", &n);
        for (int i = 1, a; i <= n; ++ i) {
            scanf("%d", &a);
            if (a & 1) odd ++;
            else even ++;
        }
        printf("%d\n", min(odd, even));
    }
}