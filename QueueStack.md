# Questions

## 155. Min Stack

用两个stack，第一个正常操作，第二个站只push比当前peek值小的

```java
class MinStack {

    Stack<Integer> s1;
    Stack<Integer> s2;
    /** initialize your data structure here. */
    public MinStack() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }

    public void push(int x) {
        s1.push(x);
        if (s2.empty() || x <= s2.peek()) {
            s2.push(x);
        }
    }

    public void pop() {
        int x = s1.pop();
        if (x == s2.peek()) {
            s2.pop();
        }
    }

    public int top() {
        return s1.peek();
    }

    public int getMin() {
        return s2.peek();
    }
}
```  

### 716. Max Stack

```java
class MaxStack {

    Stack<Integer> stack;
    Stack<Integer> maxStack;
    /** initialize your data structure here. */
    public MaxStack() {
        stack = new Stack<>();
        maxStack = new Stack<>();
    }

    public void push(int x) {
        pushHelper(x);
    }

    public void pushHelper(int x) {
        int tempMax = maxStack.isEmpty() ? Integer.MIN_VALUE : maxStack.peek();
        if (x > tempMax) {
            tempMax = x;
        }
        stack.push(x);
        maxStack.push(tempMax);
    }

    public int pop() {
        maxStack.pop();
        return stack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int peekMax() {
        return maxStack.peek();
    }

    public int popMax() {
        int max = maxStack.peek();
        Stack<Integer> temp = new Stack<>();

        // // make sure to pop all the element in the maxStack which equals to max
        while (stack.peek() != max) {
            temp.push(stack.pop()); // store smaller item inside the stack temporarly
            maxStack.pop();
        }
        stack.pop();
        maxStack.pop();
        while (!temp.isEmpty()) { // push temp stack element back into stack
            int x = temp.pop();
            pushHelper(x);
        }
        return max;
    }
}
```

### 394. Decode String

用两个stack, 一个记录数字，一个记录内容，遍历字符串，分四种情况考虑

```java
public String decodeString(String s) {
    String res = "";
    Stack<Integer> count = new Stack<>(); // 存储字符串重复次数
    Stack<String> str = new Stack<>(); // 存储中间结果
    int i = 0;
    while (i < s.length()) {
        if (Character.isDigit(s.charAt(i))) { // 读取完数字存入count
            int cnt = 0;
            while (Character.isDigit(s.charAt(i))) {
                cnt = cnt * 10 + (s.charAt(i) - '0');
                i++;
            }
            count.push(cnt);
        } else if (s.charAt(i) == '[') { // 将'['前面的结果放入str作为中间结果
            str.push(res);
            res = "";
            i++;
        } else if (s.charAt(i) == ']') { // 取出最近的n[str]，将当前[]内的结果append上
            StringBuilder sb = new StringBuilder(str.pop());
            int rep = count.pop();
            for (int j = 0; j < rep; j++) {
                sb.append(res);
            }
            res = sb.toString();
            i++;
        } else {
            res += s.charAt(i);
            i++;
        }
    }
    return res;
}
```  

### 341. Flatten Nested List Iterator

由于栈的后进先出的特性，我们在对向量遍历的时候，从后往前把对象压入栈中，那么第一个对象最后压入栈就会第一个取出来处理，  
我们的hasNext()函数需要遍历栈，并进行处理，如果栈顶元素是整数，直接返回true，如果不是，那么移除栈顶元素，  
并开始遍历这个取出的list，还是从后往前压入栈，循环停止条件是栈为空，返回false

```java
public class NestedIterator implements Iterator<Integer> {

    Stack<NestedInteger> stack;
    public NestedIterator(List<NestedInteger> nestedList) {
        stack = new Stack<>();
        for (int i = nestedList.size() - 1; i >= 0; i--) {
            stack.push(nestedList.get(i));
        }
    }

    @Override
    public Integer next() {
        return stack.pop().getInteger();
    }

    @Override
    public boolean hasNext() {
        while (!stack.empty()) {
            NestedInteger ni = stack.peek();
            if (ni.isInteger()) {
                return true;
            }
            stack.pop();
            for (int i = ni.getList().size() - 1; i >= 0; i--) {
                stack.push(ni.getList().get(i));
            }
        }
        return false;
    }
}
```  

### 295. Find Median from Data Stream

使用两个pq，small定义为max heap，large定义为min heap，addNum为O(logn), findMedian()为O(1)

```java
class MedianFinder {

    PriorityQueue<Integer> small;
    PriorityQueue<Integer> large;
    boolean even = true;
    /** initialize your data structure here. */
    public MedianFinder() {
        small = new PriorityQueue<>((a, b) -> a - b); // max heap
        large = new PriorityQueue<>((a, b) -> b - a); // min heap
    }

    public void addNum(int num) {
        if (even) { // 永远把奇数情况下的中间值放在small中
            large.offer(num);
            small.offer(large.poll());
        } else {
            small.offer(num);
            large.offer(small.poll());
        }
        even = !even;
    }

    public double findMedian() {
        if (even) {
            return (large.peek() + small.peek()) / 2.0; // use 2.0
        } else {
            return small.peek();
        }
    }
}
```  

### 75. Sort Colors

1. Two passes, use hashmap记录每个颜色出现的次数，overwrite原来的数组, counting sort

```java
public void sortColors(int[] nums) {
    int[] colors = new int[3];
    for (int num : nums) {
        colors[num]++;
    }

    int index = 0;
    for (int i = 0; i < colors.length; i++) {
        for (int j = 0; j < colors[i]; j++) {
            nums[index] = i;
            index++;
        }
    }
}
```  

1. One pass, quicksort 3 way partition.

- 定义red指针指向开头位置，blue指针指向末尾位置。  
- 从头开始遍历原数组，如果遇到0，则交换该值和red指针指向的值，并将red指针后移一位。若遇到2，则交换该值和blue指针指向的值，并将blue指针前移一位。若遇到1，则继续遍历。  

```java
public void sortColors(int[] nums) {
    int start = 0;
    int end = nums.length - 1;
    int i = 0;
    while (i <= end) {
        if (nums[i] == 0) { // make sure i is in front of start, in case [0,0,0,0,0...]
            swap(nums, i, start);
            start++;
            i++;
        } else if (nums[i] == 2) {
            swap(nums, i, end);
            end--;
        } else {
            i++;
        }
    }
}

public void swap(int[] nums, int i, int j) {
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}
```  

## BFS

## DFS

### 79. Word Search

思路： DFS，遍历board，找到和word第一个字母相同时开始dfs。

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        if (word == null || word.length() == 0 || board.length == 0) {
            return true;
        }
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == word.charAt(0) && dfs(board, word, visited, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, boolean[][] visited,
                        int i, int j, int idx) {
        // the last character match
        if (idx == word.length()) {
            return true;
        }
        // index out of bound, character doesn't match or been visited
        if (i < 0 || i >= board.length
            || j < 0 || j >= board[i].length
            || board[i][j] != word.charAt(idx)
            || visited[i][j]) {
            return false;
        }
        visited[i][j] = true;
        // dfs all four directions
        if (dfs(board, word, visited, i + 1, j, idx + 1)
           || dfs(board, word, visited, i - 1, j, idx + 1)
           || dfs(board, word, visited, i, j + 1, idx + 1)
           || dfs(board, word, visited, i, j - 1, idx + 1)) {
            return true;
        }
        // backtracking, reset visited point
        visited[i][j] = false;
        return false;
    }
}
```  

### 212. Word Search II

```java
// 第一个思路，每个单词都用dfs在matrix中查找，O(MNN)
// 用trie，O(MNLogN)
class TrieNode {
    TrieNode[] children;
    String word;

    public TrieNode() {
        // a-z, all lower case
        children = new TrieNode[26];
    }
}
class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        // 建树，将单词存到树的叶子节点
        TrieNode root = buildTrie(words);
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, i, j, root, res);
            }
        }
        return res;
    }

    public void dfs(char[][] board, int i, int j, TrieNode runner, List<String> res) {
        char c = board[i][j]; // for backtracking
        if (c == '#' || runner.children[c - 'a'] == null) {
            return;
        }
        runner = runner.children[c - 'a'];
        if (runner.word != null) {
            res.add(runner.word);
            runner.word = null; // 将当前单词设为null，防止相同结果出现
        }
        board[i][j] = '#'; // 标记为当前单词已用过的点
        // 提前检查边界，减少递归调用
        if (i > 0) {
            dfs(board, i - 1, j, runner, res);
        }
        if (j > 0) {
            dfs(board, i, j - 1, runner, res);
        }
        if (i < board.length - 1) {
            dfs(board, i + 1, j, runner, res);
        }
        if (j < board[0].length - 1) {
            dfs(board, i, j + 1, runner, res);
        }
        board[i][j] = c;
    }

    public TrieNode buildTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode runner = root;
            for (char c : word.toCharArray()) {
                if (runner.children[c - 'a'] == null) {
                    runner.children[c - 'a'] = new TrieNode();
                }
                runner = runner.children[c - 'a'];
            }
            runner.word = word;
        }
        return root;
    }
}
```  
