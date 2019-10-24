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

## BFS

## DFS
