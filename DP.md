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

## 322. Coin Change

DP solution, think in bottom-up manner. Suppose we have already computed all the minimum counts up to sum, what would be the minimum count for sum+1?

```java
public int coinChange(int[] coins, int amount) {
    if (amount == 0) return 0;
    int[] dp = new int[amount + 1];
    int sum = 1;
    while (sum <= amount) {
        int min = -1;
        for (int coin : coins) { // 寻找可以组合成amount的组合
            if (sum >= coin && dp[sum - coin] != -1) { // amount还有剩余并且剩余的可以组合起来(!= -1)
                int tmp = dp[sum - coin] + 1; // 所需硬币数量+1
                if (min < 0 || tmp < min) { // 只有在min还未更新或者tmp<min下才更新min
                    min = tmp;
                }
            }
        }
        dp[sum] = min; // 更新sum所需的最少硬币
        sum++;
    }
    return dp[amount];
}
```  

## 221. Maximal Square

dp[i][j] 代表在以i, j这一格为右下角的正方形边长。如果这一格的值也是1，那这个正方形的边长就是他的上面，左手边，和斜上的值的最小边长 +1。因为如果有一边短了缺了，都构成不了正方形。

```java
public int maximalSquare(char[][] matrix) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
        return 0;
    }
    int res = 0;
    int n = matrix.length;
    int m = matrix[0].length;
    // dp[i][j] represent right lower of the square the length of the square
    int[][] dp = new int[n + 1][m + 1];
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (matrix[i - 1][j - 1] == '1') {
                dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                res = Math.max(dp[i][j], res);
            }
        }
    }
    return res * res;
}
```  

## 256. Paint House

```java
class Solution {
    public int minCost(int[][] costs) {
        int n = costs.length;
        if (n == 0) {
            return 0;
        }
        int[][] dp = new int[n][3];
        for (int i = 0; i < 3; i++) {
            dp[0][i] = costs[0][i];
        }

        for (int i = 1; i < n; i++) {
            dp[i][0] = costs[i][0] + Math.min(dp[i - 1][1], dp[i - 1][2]);
            dp[i][1] = costs[i][1] + Math.min(dp[i - 1][0], dp[i - 1][2]);
            dp[i][2] = costs[i][2] + Math.min(dp[i - 1][1], dp[i - 1][0]);
        }
        return Math.min(dp[n - 1][0], Math.min(dp[n - 1][1], dp[n - 1][2]));
    }
}
```  

## 198. House Robber

Recursion with memo, O(n) time and space

```java
class Solution {
    int[] memo;
    public int rob(int[] nums) {
        memo = new int[nums.length + 1];
        Arrays.fill(memo, -1);
        return helper(nums, nums.length - 1);
    }

    private int helper(int[] nums, int i) {
        if (i < 0) {
            return 0;
        }
        if (memo[i] >= 0) {
            return memo[i];
        }
        int res = Math.max(helper(nums, i - 2) + nums[i], helper(nums, i - 1));
        memo[i] = res;
        return res;
    }
}
```

DP solution

```java
public int rob(int[] nums) {
    if (nums.length == 0) return 0;
    int[] memo = new int[nums.length + 1];
    memo[0] = 0;
    memo[1] = nums[0];
    for (int i = 1; i < nums.length; i++) {
        int val = nums[i];
        // memo[i]，不抢当前的房子
        // memo[i - 1] + val，抢当前的房子加上向前两个房子的抢钱总额
        memo[i + 1] = Math.max(memo[i - 1] + val, memo[i]);
    }
    return memo[nums.length];
}
```  

## 1048. Longest String Chain

```java
// 对数组按照长度排序
// 遍历每一个词，将该词删掉一个字符后的所有情况进行检查
// 如果删除字符后的词出现过，更新该字符对应map里的值
// 每次检查后更新返回值
class Solution {
    public int longestStrChain(String[] words) {
        Map<String, Integer> map = new HashMap<>();
        // 按照字符串长度排序
        Arrays.sort(words, (a, b) -> a.length() - b.length());
        int res = 0;
        for (String word : words) {
            int len = 0;
            for (int i = 0; i < word.length(); i++) {
                // 每次检查去掉字符后的字符串
                String prev = word.substring(0, i) + word.substring(i + 1);
                len = Math.max(len, map.getOrDefault(prev, 0) + 1);
            }
            map.put(word, len);
            res = Math.max(res, len);
        }
        return res;
    }
}
```  

## 312. Burst Balloons

```java
// Divide & Conquer
class Solution {
    public int maxCoins(int[] nums) {
        int len = nums.length;
        int[] balloons = new int[len + 2];
        balloons[0] = 1;
        balloons[len + 1] = 1;
        for (int i = 1; i < len + 1; i++) { // 将数组的两端填充上1
            balloons[i] = nums[i - 1];
        }

        int[][] dp = new int[len + 2][len + 2];
        helper(balloons, dp, 1, len);
        return dp[1][len];
    }

    private int helper(int[] nums, int[][] dp, int start, int end) {
        if (start > end) {
            return 0;
        }
        if (dp[start][end] > 0) {
            return dp[start][end];
        }
        for (int i = start; i <= end; i++) {
            int value = nums[start - 1] * nums[i] * nums[end + 1]
                + helper(nums, dp, start, i - 1) + helper(nums, dp, i + 1, end);
            dp[start][end] = Math.max(dp[start][end], value);
        }
        return dp[start][end];
    }
}

// DP
// Time Complexity: O(N^3)
// Space Complexity: O(N^2)
class Solution {
    public int maxCoins(int[] nums) {
        int len = nums.length;
        int[] balloons = new int[len + 2];
        balloons[0] = 1;
        balloons[len + 1] = 1;
        for (int i = 1; i < len + 1; i++) {
            balloons[i] = nums[i - 1];
        }

        int[][] dp = new int[len + 2][len + 2];
        for (int n = 1; n <= len; n++) { // 打爆气球的个数
            for (int i = 1; i <= len - n + 1; i++) {
                int j = i + n - 1;
                for (int k = i; k <= j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][k - 1] + dp[k + 1][j] + balloons[i - 1]*balloons[k]*balloons[j + 1]);
                }
            }
        }
        return dp[1][len];
    }
}
```  
