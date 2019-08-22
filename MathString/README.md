### 20. Valid Parentheses
使用stack, 遍历的时候将对称的parentheses放入stack，不存在的时候pop，最后检查stack是否为空
```java
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

### 937. Reorder Log Files
用两个ArrayList分别存num log和letter log, sort letter log, 最后拼在一起
```java
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

### 7. Reverse Integer
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

### 344. Reverse String
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

### 13. Roman to Integer
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

### 202. Happy Number
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
2. HashSet, 如果新的计算结果不能加到set中，则返回false，只有在计算结果为1时返回true
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

### 415. Add Strings
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

### 5. Longest Palindrome Substring
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

### 22. Generate Parentheses
First, the first character should be “(“. Second, at each step, you can either print “(“ or “)”,  
but print “)” only when there are more “(“s than “)”s. Stop printing out “(“ when the number of “(“ s hit n.  
The first actually merges into the second condition.
```java
public List<String> generateParenthesis(int n) {
    List<String> list = new ArrayList<>();
    backtracking(list, "", 0, 0, n);
    return list;
}

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

### 17. Letter Combination of a Phone Number
1. Single queue BFS
public List<String> letterCombinations(String digits) {
    LinkedList<String> list = new LinkedList<>();
    if (digits.isEmpty()) {
        return list;
    }

    String[] buttons = {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    list.add("");
    char[] dc = digits.toCharArray();

    for (int i = 0; i < dc.length; i++) {
        char[] letters = buttons[dc[i] - '0'].toCharArray();
        while (list.peek().length() == i) { // Make sure it's on the same level
            String row = list.poll();
            for (char letter : letters) {
                list.add(row + letter);
            }
        }
     }
    return list;
}
```  

### 273. Integer to English Word
Intuitive
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

### 227. Basic Calculator II
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

### 8. String to Integer (atoi)
```java
public int myAtoi(String str) {
    int index = 0, sign = 1, total = 0;
    //1. Empty string
    if(str.length() == 0) {
        return 0;
    }

    //2. Remove Spaces
    while(index < str.length() && str.charAt(index) == ' ') {
        index ++;
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
        index ++;
    }
    return total * sign;
}
```  

## Hard  
### 10. Regular Expression Matching
1. Funny solution
```java
public boolean isMatch(String s, String p) {
    return s.matches(p);
}
```  
2. 1, If p.charAt(j) == s.charAt(i) :  dp[i][j] = dp[i-1][j-1];
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

### 68. Text Justification
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

### 50. Pow(x,n)
用二分法，注意int越界情况
```java
public double myPow(double x, int n) {
    double ans;
    if (n < 0) {
        x = 1 / x;
        n = -(n + 1); // 处理int越界情况
        ans = 1.0 * x;
    } else {
        ans = 1.0;
    }
    
    double tmp = x;
    while (n != 0) {
        if (n % 2 == 1) {
            ans *= tmp;
        }
        tmp *= tmp;
        n /= 2;
    }

    return ans;
}
```  

### 125. Valid Palindrome
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
