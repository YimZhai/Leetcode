# Questions

## 509. Fibonacci Number

```java
// recursion with memorization
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public int fib(int N) {
        if (map.containsKey(N)) {
            return map.get(N);
        }
        int res = 0;
        if (N < 2) {
            res = N;
        } else {
            res = fib(N - 1) + fib(N - 2);
        }
        map.put(N, res);
        return res;
    }
}
```

```java
// DP
public int fib(int N) {
    if (N == 0) {
        return 0;
    }
    if (N == 1) {
        return 1;
    }
    int[] f = new int[N + 1];
    f[0] = 0;
    f[1] = 1;
    for (int i = 2; i <= N; i++) {
        f[i] = f[i - 1] + f[i - 2];
    }
    return f[N];
}
```  

## 70. Climbing Stairs

```java
// recursion with memorization
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public int climbStairs(int n) {
        if (map.containsKey(n)) {
            return map.get(n);
        }
        int res = 0;
        if (n < 2) {
            res = 1;
        } else {
            res = climbStairs(n - 1) + climbStairs(n - 2);
        }
        map.put(n, res);
        return res;
    }
}
```

```java
// dp
public int climbStairs(int n) {
    if (n <= 1) {
        return 1;
    }
    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
        // 第n层是从n-1层爬一层，或者n-2层爬两层
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```  

## 746. Min Cost Climbing Stairs

DP solution, O(n) space and time

```java
public int minCostClimbingStairs(int[] cost) {
    int len = cost.length;
    int[] dp = new int[len + 1];
    dp[0] = cost[0];
    dp[1] = cost[1];
    for (int i = 2; i <= len; i++) {
        int c = (i == len) ? 0 : cost[i];
        dp[i] = Math.min((dp[i - 2] + c), (dp[i - 1] + c));
    }
    return dp[len];
}
```

O(1) space

```java
public int minCostClimbingStairs(int[] cost) {
    int len = cost.length;
    for (int i = 2; i < len; i++) {
        cost[i] += Math.min(cost[i - 2], cost[i - 1]);
    }
    return Math.min(cost[len - 1], cost[len - 2]);
}
```

## 62. Unique Paths

```java
// DP, O(mn) time, O(mn) space
public int uniquePaths(int m, int n) {
    int[][] map = new int[m][n];
    for (int i = 0; i < m; i++) {
        map[i][0] = 1;
    }
    for (int j = 0; j < n; j++) {
        map[0][j] = 1;
    }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            map[i][j] = map[i - 1][j] + map[i][j - 1];
        }
    }
    return map[m - 1][n - 1];
}
```

## 91. Decode Ways

```java
/* DP solution, O(N) time and space
    dp[i]: ways to decode s[0] - s[i]
    dp[i] =
        ans = 0, if s[i], s[i-1]s[i] invalid
        dp[i - 1] + dp[i - 2], if both are valid
        dp[i - 1], if s[i] valid
        dp[i - 2], if s[i-1]s[i] valid
    s[i] valid, s[i] != 0;
    s[i-1]s[i] valid, 10 <= s[i-1]s[i] <=26
    S = "102213"
*/
/*  i    s[i-1]      s[i-2]s[i-1]    dp[i]
    0     N/A           N/A             1
    1       1           N/A             1
    2     N/A           10              1
    3       2           N/A             1
    4       2           22              2
    5       1           21              3
    6       3           13              5
*/
public int numDecodings(String s) {
    // corner case
    if (s == null || s.length() == 0) {
        return 0;
    }
    if (s.charAt(0) == '0') {
        return 0;
    }
    if (s.length() == 1) {
        return 1;
    }
    int[] dp = new int[s.length() + 1];
    dp[0] = 1; // empty string has 1 way
    dp[1] = 1; // one string has 1 way
    // space optimization
    /*
    int w1 = 1;
    int w2 = 1;
    */
    int tmp;
    for (int i = 2; i <= s.length(); i++) {
        // optimization
        // int w = 0;
        //检查当前字符是不是'0'
        tmp = Integer.parseInt(s.substring(i - 1, i));
        if (tmp != 0) {
            dp[i] += dp[i - 1];
            // w += w1;
        }
        // 检查当前字符和前一个字符组合在一起是否在10-26之间
        if (s.charAt(i - 2) != '0') {
            tmp = Integer.parseInt(s.substring(i - 2, i));
            if (tmp >= 10 && tmp <= 26) {
                dp[i] += dp[i - 2];
                // w += w2;
            }
        }
        // w2 = w1;
        // w1 = w;
    }
    return dp[s.length()];
    // return w1;
}
```
