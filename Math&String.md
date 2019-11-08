# Questions

## 118. Pascal's Triangle

```java
// intuitive solution
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 1; i <= numRows; i++) {
            List<Integer> list = new ArrayList<>();
            for (int j = 0; j < i; j++) {
                if (j == 0) {
                    list.add(1);
                } else if (j == i - 1) {
                    list.add(1);
                } else if (j > 0 && i > 1) {
                    list.add(res.get(i - 2).get(j) + res.get(i - 2).get(j - 1));
                }
            }
            res.add(list);
        }
        return res;
    }
}
```

```java
// recursive
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        helper(res, numRows);
        return res;
    }

    public void helper(List<List<Integer>> res, int rows) {
        if (rows == 1) {
            res.add(Arrays.asList(1));
        } else if (rows > 1) {
            helper(res, rows - 1);
            List<Integer> previous = res.get(rows - 2);
            List<Integer> list = new ArrayList<>();
            int n = previous.size();
            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    list.add(1);
                }
                if (i > 0) {
                    list.add(previous.get(i) + previous.get(i - 1));
                }
                if (i == n - 1) {
                    list.add(1);
                }
            }
            res.add(list);
        }
    }
}
```  

## 119. Pascal's Triangle II

沿用上一题解法，计算到k层的时候返回结果，time, O(N), space, O(K^2)
O(K) space solution

```java
// O(K^2) time
class Solution {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        res.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            for (int j = i - 1; j >= 1; j--) {
                int sum = res.get(j) + res.get(j - 1);
                res.set(j, sum);
            }
            res.add(1);
        }
        return res;
    }
}
```

## 20. Valid Parentheses

```java
// 使用stack, 遍历的时候将对称的parentheses放入stack，不存在的时候pop，最后检查stack是否为空
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    char[] chs = s.toCharArray();
    for (char c : chs) {
        if (c == '{') {
            stack.push('}');
        } else if (c == '(') {
            stack.push(')');
        } else if (c == '[') {
            stack.push(']');
        } else {
            if (stack.empty() || stack.pop() != c) {
                return false;
            }
        }
    }
    return stack.empty();
}
```  

## 937. Reorder Log Files

```java
// 用两个ArrayList分别存num log和letter log, sort letter log, 最后拼在一起
class LetterComparator implements Comparator<String> {
    @Override
    public int compare(String s1, String s2) {
        String ss1 = s1.substring(s1.indexOf(" ") + 1);
        String ss2 = s2.substring(s2.indexOf(" ") + 1);
        if (ss1.compareTo(ss2) > 0) {
            return 1;
        } else {
            return -1;
        }
    }
}

class Solution {
    public String[] reorderLogFiles(String[] logs) {
        List<String> numList = new ArrayList<>();
        List<String> letterList = new ArrayList<>();

        for (String log : logs) {
            if (Character.isDigit(log.charAt(log.length() - 1))) {
                numList.add(log);
            } else {
                letterList.add(log);
            }
        }

        Collections.sort(letterList, new LetterComparator());
        String[] res = new String[logs.length];
        letterList.addAll(numList);
        res = letterList.toArray(res);
        return res;
    }
}
```  

## 7. Reverse Integer

Use Long for result incase of overflow

```java
public int reverse(int x) {
    long res = 0;
    while (x != 0) {
        res = res * 10 + x % 10;
        x /= 10;
    }
    if (res > Integer.MAX_VALUE || res < Integer.MIN_VALUE) {
        return 0;
    }
    return (int)res;
}
```

## 344. Reverse String

Two Pointer, left, right

```java
public void reverseString(char[] s) {
    int l = 0;
    int r = s.length - 1;
    while (l < r) {
        char tmp = s[l];
        s[l] = s[r];
        s[r] = tmp;
        l++;
        r--;
    }
}
```  

```java
// resursive
class Solution {
    public void reverseString(char[] s) {
        int n = s.length;
        helper(s, 0, n - 1);
    }

    public void helper(char[] s, int left, int right) {
        if (left > right) {
            return;
        }
        char c = s[left];
        s[left] = s[right];
        s[right] = c;
        helper(s, left + 1, right - 1);
    }
}
```  

## 151. Reverse Words in a String

```java
// just for reference
class Solution {
    public String reverseWords(String s) {
    if (s == null) return null;
    char[] a = s.toCharArray();
    int n = a.length;
    // reverse the whole string
    reverse(a, 0, n - 1);
    // reverse each word
    reverseWords(a, n);
    // clean up spaces
    return cleanSpaces(a, n);
  }
  
  void reverseWords(char[] a, int n) {
    int i = 0, j = 0;
    while (i < n) {
      while (i < j || i < n && a[i] == ' ') i++; // skip spaces
      while (j < i || j < n && a[j] != ' ') j++; // skip non spaces
      reverse(a, i, j - 1);                      // reverse the word
    }
  }
  
  // trim leading, trailing and multiple spaces
  String cleanSpaces(char[] a, int n) {
    int i = 0, j = 0;
    while (j < n) {
      while (j < n && a[j] == ' ') j++;             // skip spaces
      while (j < n && a[j] != ' ') a[i++] = a[j++]; // keep non spaces
      while (j < n && a[j] == ' ') j++;             // skip spaces
      if (j < n) a[i++] = ' ';                      // keep only one space
    }
  
    return new String(a).substring(0, i);
  }

  private void reverse(char[] a, int i, int j) {
    while (i < j) {
      char t = a[i];
      a[i++] = a[j];
      a[j--] = t;
    }
  }
}
```  

## 186. Reverse Words in a String II

```java
// 先reverse整个array，再分别reverse每个词
class Solution {
    public void reverseWords(char[] s) {
        // first reverse the whole string
        reverse(s, 0, s.length - 1);

        int right = 0;
        while (right < s.length) {
            int left = right;
            // find the right index of each word
            while (right < s.length && s[right] != ' ') {
                right++;
            }
            reverse(s, left, right - 1); // reverse each word
            right++;
        }
    }

    private void reverse(char[] s, int left, int right) {
        while (left < right) {
            char tmp = s[left];
            s[left] = s[right];
            s[right] = tmp;
            left++;
            right--;
        }
    }
}
```  

## 557. Reverse Words in a String III

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }
}
```

## 13. Roman to Integer

```java
public int romanToInt(String s) {
    char[] chs = s.toCharArray();
    Map<Character, Integer> map = new HashMap<>();
    map.put('I', 1);
    map.put('V', 5);
    map.put('X', 10);
    map.put('L', 50);
    map.put('C', 100);
    map.put('D', 500);
    map.put('M', 1000);
    int sum = 0;
    for (int i = 0; i < chs.length; i++) {
        if (i == (chs.length - 1)) {
            sum += map.get(chs[i]);
            continue;
        }
        if (map.get(chs[i]) < map.get(chs[i+1])) {
            sum -= map.get(chs[i]);
        } else {
            sum += map.get(chs[i]);
        }
    }
    return sum;
}
```  

## 829. Consecutive Numbers Sum

```java
class Solution {
    public int consecutiveNumbersSum(int N) {
        int res = 0;
        int m = 1;
        while (true) {
            int mx = N - m * (m - 1) / 2;
            if (mx <= 0) {
                break;
            }
            if (mx % m == 0) {
                res++;
            }
            m++;
        }
        return res;
    }
}
```  

## 204. Count Primes

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

## 202. Happy Number

1. Floyed Cycle Detection, O(1) space

```java
private int calSum(int n) {
    int sum = 0, tmp;
    while (n != 0) {
        tmp = n % 10;
        sum += tmp * tmp;
        n /= 10;
    }
    return sum;
}

public boolean isHappy(int n) {
    int slow, fast;
    slow = fast = n;
    while (slow > 1) {
        slow = calSum(slow);
        if (slow == 1) {
            return true;
        }
        fast = calSum(fast);
        fast = calSum(fast);
        if (fast == 1) {
            return true;
        }
        if (slow == fast) { // loop detected
            return false;
        }
    }
    return true;
}
```  

2.HashSet, 如果新的计算结果不能加到set中，则返回false，只有在计算结果为1时返回true

```java
public boolean isHappy(int n) {
    Set<Integer> set = new HashSet<>();
    int tmp, sum;
    while (set.add(n)) {
        sum = 0;
        while (n > 0) {
            tmp = n % 10;
            sum += tmp * tmp;
            n /= 10;
        }
        if (sum == 1) {
            return true;
        } else {
            n = sum;
        }
    }
    return false;
}
```  

## 28. Implement strStr()

```java
// Traditional, O(N^2) worset case, O(1) space
class Solution {
    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) {
            return 0;
        }
        if (haystack.length() == 0 || haystack.length() < needle.length()) {
            return -1;
        }
        for (int i = 0; i < haystack.length() - needle.length() + 1; i++) {
            int j;
            for (j = 0; j < needle.length(); j++) {
                if (haystack.charAt(i + j) != needle.charAt(j)) {
                    break;
                }
            }
            if (j == needle.length()) {
                return i;
            }
        }
        return -1;
    }
}

// faster version
class Solution {
    public int strStr(String haystack, String needle) {
        int n = needle.length();
        int m = haystack.length();
        if (n == 0) {
            return 0;
        }
        if (n > m) {
            return -1;
        }
        for (int i = 0; i <= m - n; i++) {
            if (haystack.charAt(i) == needle.charAt(0)) {
                if (haystack.substring(i, i + n).equals(needle)) {
                    return i;
                }
            }
        }
        return -1;
    }
}
```  

## 14. Longest Common Prefix

```java
// shorter version
// O(NM), N is the length of strs, M is the average length of each str
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        String pre = strs[0];
        for (int i = 1; i < strs.length; i++) {
            // 若pre不是当前词的前缀，每次删除pre的最后一个字符
            // 直到pre成为当前词的前缀
            while (strs[i].indexOf(pre) != 0) {
                pre = pre.substring(0, pre.length() - 1);
            }
        }
        return pre;
    }
}
// Also can use Trie tree for this problem
```  

## 415. Add Strings

用StringBuilder，从末位开始向前叠加，使用carry记录进位

```java
public String addStrings(String num1, String num2) {
    int i = num1.length() - 1;
    int j = num2.length() - 1;
    int carry = 0;
    char[] num1Array = num1.toCharArray();
    char[] num2Array = num2.toCharArray();
    StringBuilder sb = new StringBuilder();
    while (i >= 0 || j >= 0 || carry == 1) {
        int a = i >= 0 ? (num1Array[i--] - '0') : 0;
        int b = j >= 0 ? (num2Array[j--] - '0') : 0;
        int sum = a + b + carry;
        sb.insert(0, sum % 10);
        carry = sum / 10;
    }
    return sb.toString();
}
```  

## 5. Longest Palindrome Substring

1. Brute Force O(n^3)
2. 从中心向两端扩展，奇偶分开考虑, (s, i, i), (s, i, i + 1), 找到新的len更新longest，计算出start位置

```java
public String longestPalindrome(String s) {
    if (s == null || s.length() == 0) {
        return "";
    }

    int start = 0;
    int len = 0;
    int longest = 0;
    for (int i = 0; i < s.length(); i++) {
        len = findLongest(s, i, i);
        if (len > longest) {
            longest = len;
            start = i - len / 2;
        }
        len = findLongest(s, i, i + 1);
        if (len > longest) {
            longest = len;
            start = i - len / 2 + 1;
        }
    }
    return s.substring(start, start + longest);
}

private int findLongest(String s, int start, int end) {
    int len = 0;
    while (start >= 0 && end < s.length()) {
        if (s.charAt(start) != s.charAt(end)) {
            break;
        }
        len += start == end ? 1 : 2;
        start--;
        end++;
    }
    return len;
}
```  

## 22. Generate Parentheses

First, the first character should be “(“. Second, at each step, you can either print “(“ or “)”,  
but print “)” only when there are more “(“s than “)”s. Stop printing out “(“ when the number of “(“ s hit n.  
The first actually merges into the second condition.

```java
public List<String> generateParenthesis(int n) {
    List<String> list = new ArrayList<>();
    backtracking(list, "", 0, 0, n);
    return list;
}

// @Param: left, number of left brackets
// @Param: right, number of right brackets
public void backtracking(List<String> list, String str,
                         int left, int right, int n) {
    if (str.length() == 2*n) {
        list.add(str);
        return;
    }

    if (left < n) {
        backtracking(list, str + "(", left + 1, right, n);
    }
    if (right < left) {
        backtracking(list, str + ")", left, right + 1, n);
    }
}
```  

## 17. Letter Combination of a Phone Number

1. Single queue BFS

```java
public List<String> letterCombinations(String digits) {
    LinkedList<String> list = new LinkedList<>();
    if (digits.isEmpty()) {
        return list;
    }

    String[] buttons = {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    list.add("");
    char[] dc = digits.toCharArray(); // 2, 3

    for (int i = 0; i < dc.length; i++) {
        char[] letters = buttons[dc[i] - '0'].toCharArray(); // a, b, c
        while (list.peek().length() == i) { // Make sure it's on the same level
            String row = list.poll();
            for (char letter : letters) { // "" + a, "" + b, "" + c
                list.add(row + letter);
            }
        }
     }
    return list;
}
```  

## 273. Integer to English Word

```java
public String numberToWords(int num) {
    if (num == 0) {
        return "Zero";
    }
    return helper(num);
}

public String helper(int num) {
    String[] words = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen",
    "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    StringBuilder res = new StringBuilder();
    if (num >= 1000000000) {
        res.append(helper(num / 1000000000)).append(" Billion ").append(helper(num % 1000000000));
    } else if (num >= 1000000) {
        res.append(helper(num / 1000000)).append(" Million ").append(helper(num % 1000000));
    } else if (num >= 1000) {
        res.append(helper(num / 1000)).append(" Thousand ").append(helper(num % 1000));
    } else if (num >= 100) {
        res.append(helper(num / 100)).append(" Hundred ").append(helper(num % 100));
    } else if (num >= 20) {
        res.append(words[(num - 20) / 10 + 20]).append(" ").append(helper(num % 10));
    } else {
        res.append(words[num]);
    }
    return res.toString().trim();
}
```  

## 953. Verify an Alien Dictionary

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

## 224. Basic Calculator

使用stack存放sign和中间结果

```java
public int calculate(String s) {
    int len = s.length();
    if (s == null || len == 0) {
        return 0;
    }
    Stack<Integer> stack = new Stack<>();
    int res = 0;
    int sign = 1;
    for (int i = 0; i < len; i++) {
        if (Character.isDigit(s.charAt(i))) { // 找到数字
            int sum = s.charAt(i) - '0';
            while (i + 1 < len && Character.isDigit(s.charAt(i + 1))) {
                sum = sum * 10 + s.charAt(i + 1) - '0';
                i++;
            }
            res += sum * sign; // 计算出中间结果
        } else if (s.charAt(i) == '+') {
            sign = 1;
        }
        else if (s.charAt(i) == '-') {
            sign = -1;
        }
        else if (s.charAt(i) == '(') { // 中间结果、符号放入栈，重置两个变量
            stack.push(res);
            stack.push(sign);
            res = 0;
            sign = 1;
        } else if (s.charAt(i) == ')') {
            res = res * stack.pop() + stack.pop();
        }
    }
    return res;
}
```  

## 227. Basic Calculator II

使用stack, 将每次计算出的数存在stack中

```java
public int calculate(String s) {
    int len = s.length();
    if (s == null || len == 0) {
        return 0;
    }

    Stack<Integer> stack = new Stack<>();
    int num = 0; // 记录每一个数字
    char sign = '+'; // 默认符号为+
    for (int i = 0; i < len; i++) {
        if (Character.isDigit(s.charAt(i))) {
            num = num * 10 + (s.charAt(i) - '0');
        }
        if (!Character.isDigit(s.charAt(i)) && (s.charAt(i) != ' ') || (i == len - 1)) {
            switch(sign) {
                case '-':
                    stack.push(-num);
                    break;
                case '+':
                    stack.push(num);
                    break;
                case '*':
                    stack.push(stack.pop() * num);
                    break;
                case '/':
                    stack.push(stack.pop() / num);
                    break;
            }
            sign = s.charAt(i);
            num = 0;
        }
    }
    int res = 0;
    for (int i : stack) {
        res += i;
    }
    return res;
}
```  

### 69. Sqrt(x)

基础办法, O(n)

```java
public int mySqrt(int x) {
    if (x < 2) return x;
    for (long i = 1; i <= x; i++) {
        if (i * i > x) {
            return (int)(i - 1);
        }
    }
    return 0;
}
```

牛顿法, O(logN)

```java
public int mySqrt(int x) {
    long r = x;
    while (r*r > x)
        r = (r + x/r) / 2;
    return (int) r;
}
```  

## 8. String to Integer (atoi)

```java
public int myAtoi(String str) {
    int index = 0, sign = 1, total = 0;
    //1. Empty string
    if(str.length() == 0) {
        return 0;
    }

    //2. Remove Spaces
    while(index < str.length() && str.charAt(index) == ' ') {
        index++;
    }

    // handle " "
    if (index == str.length()) {
        return 0;
    }
    //3. Handle signs
    if(str.charAt(index) == '+' || str.charAt(index) == '-'){
        sign = str.charAt(index) == '+' ? 1 : -1;
        index ++;
    }

    //4. Convert number and avoid overflow
    while (index < str.length()) {
        int digit = str.charAt(index) - '0';
        if (digit < 0 || digit > 9) {
            break;
        }

        //check if total will be overflow after 10 times and add digit
        if (Integer.MAX_VALUE/10 < total || Integer.MAX_VALUE/10 == total
            && Integer.MAX_VALUE %10 < digit) {
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        total = 10 * total + digit;
        index++;
    }
    return total * sign;
}
```  

## 10. Regular Expression Matching

1. Funny solution

```java
public boolean isMatch(String s, String p) {
    return s.matches(p);
}
```  

1, If p.charAt(j) == s.charAt(i) :  dp[i][j] = dp[i-1][j-1];
2, If p.charAt(j) == '.' : dp[i][j] = dp[i-1][j-1];
3, If p.charAt(j) == '*':
here are two sub conditions:
       1   if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2]  //in this case, a* only counts as empty
       2   if p.charAt(i-1) == s.charAt(i) or p.charAt(i-1) == '.':
        dp[i][j] = dp[i-1][j]    //in this case, a* counts as multiple a
        or dp[i][j] = dp[i][j-1]   // in this case, a* counts as single a
        or dp[i][j] = dp[i][j-2]   // in this case, a* counts as empty

```java
public boolean isMatch(String s, String p) {
    if(s == null || p == null) {
        return false;
    }
    boolean[][] state = new boolean[s.length() + 1][p.length() + 1];
    state[0][0] = true;
    // no need to initialize state[i][0] as false
    // initialize state[0][j]
    for (int j = 1; j < state[0].length; j++) {
        if (p.charAt(j - 1) == '*') {
            if (state[0][j - 1] || (j > 1 && state[0][j - 2])) {
                state[0][j] = true;
            }
        }
    }
    for (int i = 1; i < state.length; i++) {
        for (int j = 1; j < state[0].length; j++) {
            if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                state[i][j] = state[i - 1][j - 1];
            }
            if (p.charAt(j - 1) == '*') {
                if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                    state[i][j] = state[i][j - 2];
                } else {
                    state[i][j] = state[i - 1][j] || state[i][j - 1] || state[i][j - 2];
                }
            }
        }
    }
    return state[s.length()][p.length()];
}
```  

## 12. Integer to Roman

```java
public String intToRoman(int num) {
    if (num < 1 || num > 3999) {
        return "";
    }
    StringBuilder sb = new StringBuilder();
    int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    String[] romans = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    int i = 0;
    while (num > 0) {
        while (num >= values[i]) {
            sb.append(romans[i]);
            num -= values[i];
        }
        i++;
    }
    return String.valueOf(sb);
}
```  

## 68. Text Justification

```java
//首先要做的就是确定每一行能放下的单词数，这个不难，就是比较n个单词的长度和加上n - 1个空格的长度跟给定的长度L来比较即可
//找到了一行能放下的单词个数，然后计算出这一行存在的空格的个数，是用给定的长度L减去这一行所有单词的长度和。
//得到了空格的个数之后，就要在每个单词后面插入这些空格，这里有两种情况，比如某一行有两个单词"to" 和 "a"，给定长度L为6
//如果这行不是最后一行，那么应该输出"to   a"，如果是最后一行，则应该输出 "to a  "，所以这里需要分情况讨论，最后一行的处理方法和其他行之间略有不同。
//最后一个难点就是，如果一行有三个单词，这时候中间有两个空，如果空格数不是2的倍数，那么左边的空间里要比右边的空间里多加入一个空格，那么我们只需要用总的空格数除以空间个数
//能除尽最好，说明能平均分配，除不尽的话就多加个空格放在左边的空间里"
public List<String> fullJustify(String[] words, int maxWidth) {
    List<String> lines = new ArrayList<String>();
    int index = 0;
    while (index < words.length) {
        //count：该行所有单词累计总长度
        int count = words[index].length();
        //last:该行最后一个词的index
        int last = index + 1;
        while (last < words.length) {
            //out of bound
            if (words[last].length() + count + 1 > maxWidth) break;
            //plus one for the space, if its a perfect fit it will fit
            count += 1 + words[last].length();
            last++;
        }
        StringBuilder builder = new StringBuilder();
        //append该行第一个单词
        builder.append(words[index]);
        //这一行除去第一个已经append的单词，共剩下几个词语：diff 个：从index到last-1
        int diff = last - index - 1;
       // if last line or number of words in the line is 1, left-justified
        //最后一行：每个单词中间一个空格， 剩余补上空白
        if (last == words.length || diff == 0) {
            for (int i = index+1; i < last; i++) {
                builder.append(" ");
                builder.append(words[i]);
            }
            for (int i = builder.length(); i < maxWidth; i++) {
                builder.append(" ");
            }
        } else {
            //不是最后一行：middle justified
            //这一行总space的个数：（长度-累计单词总长度）
            //每个单词后面space的个数：（长度-累计单词总长度）/单词个数
            // r为需要平均分配到中间的空格总数
            int spaces = (maxWidth - count) / diff;
            int r = (maxWidth - count) % diff;
            for (int i = index+1; i < last; i++) {
                for(int k=spaces; k > 0; k--) {
                    builder.append(" ");
                }
                if(r > 0) {
                    builder.append(" ");
                    r--;
                }
                builder.append(" ");
                builder.append(words[i]);
            }
        }
        lines.add(builder.toString());
        index = last;
    }
    return lines;
}
```  

## 50. Pow(x,n)

用二分法，注意int越界情况

```java
class Solution {
    public double myPow(double x, int n) {
        double res = 1.0;
        if (n < 0) {
            x = 1 / x;
            n = -(n + 1); // 处理边界
            res *= x;
        } else {
            res = 1.0;
        }
        double tmp = x;
        while (n != 0) {
            if (n % 2 == 1) {
                res *= tmp;
            }
            tmp *= tmp;
            n /= 2;
        }
        return res;
    }
}
```  

## 779. K-th Symbol in Grammar

```java
// 比如我们求第11行的1000个数字是0还是1
// 那么第11行的第1000个数字 a 是根据哪儿来的呢，
// 是依照第10行的第500个数字 b 而来
// b 又是依照 9行250个数字 c 来，数字才依照 8 行 125数字 d 来， d 依照 7 行 63 e 来... 以此类推
// O(N)
class Solution {
    public int kthGrammar(int N, int K) {
        if (N == 1) {
            return 0;
        }
        if (K % 2 == 1) {
            return kthGrammar(N - 1, (K + 1) / 2) == 0 ? 0 : 1;
        } else {
            return kthGrammar(N - 1, K / 2) == 0 ? 1 : 0;
        }
    }
}
```

```java
// trick solution, O(logN)
class Solution {
    public int kthGrammar(int N, int K) {
        return Integer.bitCount(K - 1) % 2;
    }
}
```  

### 6. ZigZag Conversion

定义一个flag控制方向，定义一个数组存值，最终遍历两边字符串，O(n) time, O(n) space

```java
public String convert(String s, int numRows) {
    if (s == null || s.length() == 0) return "";
    if (numRows == 1) return s;
    char[] chs = s.toCharArray();
    List<Character>[] zig = new ArrayList[numRows];
    int idx = 0;
    boolean flag = true;
    for (int i = 0; i < s.length(); i++) {
        if (zig[idx] == null) {
            zig[idx] = new ArrayList<>();
        }
        zig[idx].add(chs[i]);
        if (flag) {
            idx++;
        } else {
            idx--;
        }
        if (idx == 0 || idx == numRows - 1) {
            flag = !flag;
        }
    }
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < numRows; i++) {
        if (zig[i] == null) continue;
        for (char c : zig[i]) {
            sb.append(c);
        }
    }
    return String.valueOf(sb);
}
```  

```java
// 另一种办法，效率高一些
public String convert(String s, int numRows) {
    char[] chs = s.toCharArray();
    int len = chs.length;
    // declare StringBuffer array
    StringBuilder[] sb = new StringBuilder[numRows];
    // initialize array
    for (int i = 0; i < numRows; i++) {
        sb[i] = new StringBuilder();
    }
    int idx = 0;
    while (idx < len) {
        for (int i = 0; i < numRows && idx < len; i++) { // top down
            sb[i].append(chs[idx++]);
        }
        for (int i = numRows - 2; i >= 1 && idx < len; i--) { // bottom up
            sb[i].append(chs[idx++]);
        }
    }
    for (int i = 1; i < numRows; i++) {
        sb[0].append(sb[i]);
    }
    return sb[0].toString();
}
```  

### 43. Multiply Strings

第一个思路是装string转换为integer，发现存在很多越界情况。  
不转换，直接通过string进行乘法操作

```java
public String multiply(String num1, String num2) {
    int n1 = num1.length();
    int n2 = num2.length();
    int[] res = new int[n1 + n2];
    for (int i = n1 - 1; i >= 0; i--) {
        for (int j = n2 - 1; j >= 0; j--) {
            int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
            int left = i + j; // 十位的下标
            int right = i + j + 1; // 个位的下标
            int sum = mul + res[right]; // 先计算是否有进位
            res[left] += sum / 10; // 更新十位
            res[right] = sum % 10; // 更新个位
        }
    }
    StringBuilder sb = new StringBuilder();
    for (int p : res) {
        if (!(sb.length() == 0 && p == 0)) { // avoid leading zero
            sb.append(p);
        }
    }
    return sb.length() == 0 ? "0" : sb.toString();
}
```  

## 125. Valid Palindrome

1. 去除空格和符号, 转为小写字母
2. 两端双指针

```java
public boolean isPalindrome(String s) {
    String input = s.replaceAll("\\p{Punct}", "").replaceAll(" ", "").toLowerCase();
    char[] charArr = input.toCharArray();
    int i = 0;
    int j = charArr.length - 1;
    while (i <= j) {
        if (charArr[i] == charArr[j]) {
            i++;
            j--;
        } else {
            return false;
        }
    }
    return true;
}
```  

## 680. Valid Palindrome II

思路同上，双指针从两端向中间移动，遇到不同的时候分情况考虑，判断左边跳一位和右边挑一位后是否还是回文

```java
class Solution {
    public boolean validPalindrome(String s) {
        char[] chs = s.toCharArray();
        int left = 0;
        int right = chs.length - 1;
        boolean jump = true;
        while (left < right) {
            if (chs[left] == chs[right]) {
                left++;
                right--;
            } else {
                return isPalindrome(chs, left + 1, right) || isPalindrome(chs, left, right - 1);
            }
        }
        return true;
    }

    public boolean isPalindrome(char[] chs, int left, int right) {
        while (left < right) {
            if (chs[left] != chs[right]) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
```  

## 647. Palindromic Substrings

```java
class Solution {
    int count = 0;
    public int countSubstrings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        for (int i = 0; i < s.length(); i++) {
            helper(s, i, i);
            helper(s, i, i + 1);
        }
        return count;
    }

    public void helper(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            count++;
            left--;
            right++;
        }
    }
}
```

## 387. First Unique Character in a String

HashMap存储每个字母出现的次数，遍历两边，寻找第一个符合要求的字母

```java
public int firstUniqChar(String s) {
    Map<Character, Integer> map = new HashMap<>();
    for (char c : s.toCharArray()) {
        map.put(c, map.getOrDefault(c, 0) + 1);
    }
    for (int i = 0; i < s.length(); i++) {
        if (map.get(s.charAt(i)) == 1) {
            return i;
        }
    }
    return -1;
}
```  

### 443. String Compression

双指针，O(n) time, O(1) space

```java
public int compress(char[] chars) {
    int cur = 0; // 赋值下标
    int index = 0; // 遍历下标
    int len = chars.length;
    while (index < len) {
        char curChar = chars[index];
        int cnt = 0;
        while (index < len && chars[index] == curChar) { // 字符相同一直遍历
            index++;
            cnt++;
        }
        chars[cur++] = curChar;
        if (cnt != 1) { // 相同字符超过1个，需要更新数组
            for (char c : Integer.toString(cnt).toCharArray()) {
                chars[cur++] = c;
            }
        }
    }
    return cur;
}
```  

## 67. Add Binary

```java
// 从最右边一位开始相加，使用一个变量记录是否进位, O(max(m, n)) time, O(max(m,n)), space
public String addBinary(String a, String b) {
    StringBuilder sb = new StringBuilder();
    int i = a.length() - 1;
    int j = b.length() - 1;
    int carry = 0;
    while (i >= 0 || j >= 0) {
        int sum = carry;
        if (i >= 0) {
            sum += a.charAt(i--) - '0';
        }
        if (j >= 0) {
            sum += b.charAt(j--) - '0';
        }
        sb.append(sum % 2);
        carry = sum / 2;
    }
    if (carry != 0) {
        sb.append(carry);
    }
    return sb.reverse().toString();
}
```  

## 1047. Remove All Adjacent Duplicates In String

```java
// use stack
class Solution {
    public String removeDuplicates(String S) {
        Stack<Character> s = new Stack<>();
        char[] chs = S.toCharArray();
        for (char c : chs) {
            if (s.empty() || s.peek() != c) {
                s.push(c);
            } else {
                s.pop();
            }
        }
        char[] res = new char[s.size()];
        for (int i = s.size() - 1; i >= 0; i--) {
            res[i] = s.pop();
        }
        return new String(res);
    }
}
```  

## 1209. Remove All Adjacent Duplicates in String II

```java
// 用两个stack, 一个存字符，一个存当前字符的个数O(N) time and space
// 优化方案，建立一个class，存储char和对应count，使用LinkedList代替stack
class Solution {
    public String removeDuplicates(String s, int k) {
        Stack<Character> sChar = new Stack<>();
        Stack<Integer> sCnt = new Stack<>();
        for (char c : s.toCharArray()) {
            // 如果stack里存在相邻相同元素
            // 更新对应的cnt
            if (!sChar.empty() && sChar.peek() == c) {
                sCnt.push(sCnt.peek() + 1);
            } else {
                sCnt.push(1);
            }
            sChar.push(c);
            if (sCnt.peek() == k) {
                for (int i = 1; i <= k; i++) {
                    sChar.pop();
                    sCnt.pop();
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!sChar.empty()) {
            sb.append(sChar.pop());
        }
        return sb.reverse().toString();
    }
}
```  

## 767. Reorganize String

```java
public String reorganizeString(String S) {
    // Count occurance of each character
    int[] hash = new int[26];
    for (char c : S.toCharArray()) {
        hash[c - 'a']++;
    }

    int len = S.length();
    int max = 0;
    int letter = 0;
    // find the letter with the largest appearance
    for (int i = 0; i < hash.length; i++) {
        if (hash[i] > max) {
            max = hash[i];
            letter = i;
        }
    }
    if (max > (len + 1) / 2) {
        return "";
    }
    // put letter into even indexes.
    char[] res = new char[len];
    int index = 0;
    while (hash[letter] > 0) {
        res[index] = (char)(letter + 'a');
        index += 2;
        hash[letter]--;
    }
    // put the rest into res
    for (int i = 0; i < hash.length; i++) {
        while (hash[i] > 0) {
            if (index >= len) {
                index = 1;
            }
            res[index] = (char)(i + 'a');
            index += 2;
            hash[i]--;
        }
    }
    return new String(res);
}
```  

## 166. Fraction to Recurring Decimal

```java
class Solution {
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        res.append((long)numerator * (long)denominator > 0 ? "" : "-");
        long num = Math.abs((long) numerator);
        long den = Math.abs((long) denominator);
        // 整数部分 2 / 3
        res.append(num / den); // 0
        num %= den; // num: 2
        if (num == 0) {
            return res.toString();
        }
        // 小数部分
        res.append(".");
        // number -> index
        Map<Long, Integer> map = new HashMap<>();
        map.put(num, res.length());
        while (num != 0) {
            num *= 10; // 20 20
            res.append(num / den); // 6 6
            num %= den; // 2
            if (map.containsKey(num)) {
                int index = map.get(num);
                res.insert(index, "(");
                res.append(")");
                break;
            } else { // 6 -> 3
                map.put(num, res.length());
            }
        }
        return res.toString();
    }
}
```  
