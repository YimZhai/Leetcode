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
