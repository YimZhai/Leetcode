### 91. Decode Ways *

DP

```java
public int numDecodings(String s) {
    if (s == null || s.length() == 0) {
        return 0;
    }

    if (s.charAt(0) == '0') {
        return 0;
    }

    int[] dp = new int[s.length() + 1];
    dp[0] = 1;
    dp[1] = 1;
    int tmp;
    for (int i = 2; i <= s.length(); i++) {
        //检查当前字符是不是'0'
        tmp = Integer.parseInt(s.substring(i - 1, i));
        if (tmp != 0) {
            dp[i] = dp[i - 1];
        }
        // 检查当前字符和前一个字符组合在一起是否在1-26之间
        if (s.charAt(i - 2) != '0') {
            tmp = Integer.parseInt(s.substring(i - 2, i));
            if (tmp > 0 && tmp <= 26) {
                dp[i] += dp[i - 2];
            }
        }

    }
    return dp[s.length()];
}
```

### 393. UTF8 Validation

知识点 Bit Manipulation [UTF8](https://www.fileformat.info/info/unicode/utf8.htm)

```java
public boolean validUtf8(int[] data) {
    int cnt = 0;
    for (int d : data) {
        if (cnt == 0) {
            if (d >> 5 == 0b110) { // 2 byte
                cnt = 1;
            } else if (d >> 4 == 0b1110) { // 3 byte
                cnt = 2;
            } else if (d >> 3 == 0b11110) { // 4 byte
                cnt = 3;
            } else if (d >> 7 != 0) { // 1 byte and not start with 0
                return false;
            }
        } else {
            if (d >> 6 != 0b10) { // check remaining byte
                return false;
            }
            cnt--;
        }
    }
    return cnt == 0; // only return true when no byte remaining.
}
```  

### 193. Valid Phone Numbers

Bash，任意xxx-开头或者(xxx) 开头

```bash
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt
```

### 322. Coin Change

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

### 221. Maximal Square

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

### 621. Task Schedular

```java
public int leastInterval(char[] tasks, int n) {
    // 统计每个字母的出现频率
    int[] counter = new int[26];
    int max = 0; // 最高频的次数
    int maxCnt = 0; // 同时出现最高频的个数
    for (char task : tasks) {
        int index = task - 'A';
        counter[index]++;
        if (max == counter[index]) {
            maxCnt++;
        } else if (max < counter[index]) {
            max = counter[index];
            maxCnt = 1;
        }
    }
    // 中间的间隔
    int partCnt = max - 1;
    // 间隔的长度
    int partLength = n - (maxCnt - 1);
    // 间隔总数
    int emptySlots = partLength * partCnt;
    // 还需要分配的任务个数
    int otherTasks = tasks.length - max * maxCnt;
    // 所需空闲数
    int idles = Math.max(0, emptySlots - otherTasks);
    // 返回任务长度加上额外的空闲数
    return tasks.length + idles;
}
```  

### 362. Design Hit Counter

输入的timestamp是单位为秒的时间, 5分钟也就是timestamp 300，使用一个队列来记录

```java
class HitCounter {

    Queue<Integer> q;
    /** Initialize your data structure here. */
    public HitCounter() {
        q = new LinkedList<>();
    }

    /** Record a hit.
        @param timestamp - The current timestamp (in seconds granularity). */
    public void hit(int timestamp) {
        q.offer(timestamp);
    }

    /** Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity). */
    public int getHits(int timestamp) {
        while (!q.isEmpty() && timestamp - q.peek() >= 300) {
            q.poll();
        }
        return q.size();
    }
}
```  

### 539. Minimum Time Difference

Bucket Sort

```java
public int findMinDifference(List<String> timePoints) {
    boolean[] time = new boolean[1440];
    for (String points : timePoints) {
        String[] t = points.split(":");
        int hour = Integer.parseInt(t[0]);
        int min = Integer.parseInt(t[1]);
        if (time[hour * 60 + min]) { // duplicate time
            return 0;
        }
        time[hour * 60 + min] = true;
    }
    int min = 1440;
    int prev = 0;
    int first = Integer.MAX_VALUE;
    int last = Integer.MIN_VALUE;
    for (int i = 0; i < 1440; i++) {
        if (time[i]) {
            if (first != Integer.MAX_VALUE) { // 排除第一个找到的时间，无法相减
                min = Math.min(min, i - prev);
            }
            first = Math.min(first, i); // 找到最小时间
            last = Math.max(last, i); // 找到最大时间
            prev = i;
        }
    }
    // corner case
    min = Math.min(min, (1440 - last + first)); // 取反向时间比较
    return min;
}
```  

### 953. Verify an Alien Dictionary

```java
public boolean isAlienSorted(String[] words, String order) {
    if (words.length == 1) return true;
    for (int i = 1; i < words.length; i++) { // O(n)判断相邻的两个词
        int j = 0;
        boolean checked = true; // 是否需要额外检查
        int len = Math.min(words[i].length(), words[i - 1].length());
        while (j < len) {
            if (order.indexOf(words[i].charAt(j)) > order.indexOf(words[i - 1].charAt(j))) { // 符合要求，不需要额外检查
                checked = false;
                break;
            } else if (order.indexOf(words[i].charAt(j)) < order.indexOf(words[i - 1].charAt(j))) { // 顺序错误返回false
                return false;
            }
            j++;
        }
        if (checked && words[i - 1].length() > words[i].length()) { // 字母顺序都一样，需要额外检查长度
            return false;
        }
    }
    return true;
}
```  

### 301. Remove Invalid Parentheses

```java
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        dfs(res, s, 0, 0, '(', ')');
        return res;
    }

    private void dfs(List<String> res, String s, int iStart, int jStart, char open, char close) {
        // 计算'('和')'的数量
        int numOpen = 0, numClose = 0;
        // 我们从iStart开始遍历, 也是在删除多余')'后开始遍历的起点
        for (int i = iStart; i < s.length(); i++) {
            // 正向操作时，只处理')'多余的情况
            if (s.charAt(i) == open) numOpen++;
            if (s.charAt(i) == close) numClose++;
            if (numClose > numOpen) {
                // 移除多余的右括号, 可能有多种移除的方案，因此要遍历所有的右括号, 此时要移除的右括号一定在i的左边
                for (int j = jStart; j <= i; j++) {
                    // 移除时要考虑相邻的右括号重复的情况, (j == jStart)为删除第一个')'的情况
                    // s.charAt(j - 1) != close 考虑当前为')'同时前一个不能为')'
                    if (s.charAt(j) == close && (j == jStart || s.charAt(j - 1) != close)) {
                        // 删除下标为j的右括号后，判断是否valid, iStart = i因为该字符串到i的位置都已经判断过为valid
                        // jStart = j表示跳过了前一层循环结束时的末尾字符
                        dfs(res, s.substring(0, j) + s.substring(j + 1, s.length()), i, j, open, close);
                    }
                }
                return; // 递归操作处理后面的字符
            }
        }
        // 正向检查过一遍，接下来倒过来检查是否有多余的'('
        String reversed = new StringBuilder(s).reverse().toString();
        if (open == '(') { // 如果是正向到了这步，说明要检查反向
            dfs(res, reversed, 0, 0, ')', '(');
        } else { // 如果是反向到了这步，则把反向翻转过来，添加到结果中
            res.add(reversed);
        }
    }
}
```  

### 198. House Robber

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

### 204. Count Primes

```java
public int countPrimes(int n) {
    boolean[] prime = new boolean[n + 1];
    int cnt = 0;
    // 从2开始，将所有质数的的倍数的值标记为true
    // 未被标记的则为质数, 如果n <= 2则返回0
    for (int i = 2; i < n; i++) {
        if (prime[i] == false) {
            cnt++;
            for (int j = 2; i * j < n; j++) {
                prime[i * j] = true;
            }
        }
    }
    return cnt;
}
```  

### 41. First Missing Integer

O(n) time and space 方法，很直观，用一个set存储出现过的值, 同时记录最大值, 然后从1开始遍历

```java
public int firstMissingPositive(int[] nums) {
    int max = 0;
    Set<Integer> set = new HashSet<>();
    for (int num : nums) {
        max = Math.max(max, num);
        set.add(num);
    }
    for (int i = 1; i <= max; i++) {
        if (!set.contains(i)) {
            return i;
        }
    }
    return max + 1;
}
```  

O(n) time and O(1) space

```java
public int firstMissingPositive(int[] nums) {
    int n = nums.length;
    for (int i = 0; i < n; i++) {
        // nums[i] 需要在数组的范围内, 同时nums[i]不在应该在的位置上
        while (nums[i] > 0 && nums[i] < n && nums[i] != nums[nums[i] - 1]) {
            int tmp = nums[i];
            nums[i] = nums[nums[i] - 1];
            nums[tmp - 1] = tmp;
        }
    }
    for (int i = 0; i < n; i++) {
        // 如果该值和下标不对应则返回
        if (nums[i] != i + 1) {
            return i + 1;
        }
    }
    return n + 1;
}
```  

### 743. Network Delay Time
