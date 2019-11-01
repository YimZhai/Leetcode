# Questions

## 206. Reversed LinkedList

```java
// iterative
public ListNode reverseList(ListNode head) {
    ListNode pre = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode node = curr.next;
        curr.next = pre;
        pre = curr;
        curr = node;
    }
    return pre;
}
```

```java
// recursive
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = reverseList(head.next); // reach to the end
        head.next.next = head;
        head.next = null;
        return node;
    }
}
```

## 24. Swap Nodes in Pairs

```java
// recursive solution, O(N) space
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode second = head.next;
        ListNode third = head.next.next;

        second.next = head;
        head.next = swapPairs(third);
        return second;
    }
}
```

```java
// iterative O(1) space
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(0);
        ListNode point = dummy;
        dummy.next = head;
        while (point.next != null && point.next.next != null) {
            ListNode n1 = point.next;
            ListNode n2 = point.next.next;
            point.next = n2;
            n1.next = n2.next;
            n2.next = n1;
            point = n1;
        }
        return dummy.next;
    }
}
```

## 21. Merge Two Sorted Lists

1. Recursion

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null) return l2;
    if (l2 == null) return l1;

    if (l1.val < l2.val) {
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;
    } else {
        l2.next = mergeTwoLists(l2.next, l1);
        return l2;
    }
}
```  

2.Iterator

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode preHead = new ListNode(-1);
    ListNode runner = preHead;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            runner.next = l1;
            l1 = l1.next;
        } else {
            runner.next = l2;
            l2 = l2.next;
        }
        runner = runner.next;
    }
    if (l1 == null) {
        runner.next = l2;
    }
    if (l2 == null) {
        runner.next = l1;
    }
    return preHead.next;
}
```  

## 2. Add Two Numbers

1. 定义carry记录进位

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode head = new ListNode(0);
    ListNode p = l1;
    ListNode q = l2;
    ListNode curr = head;
    int carry = 0;
    while (p != null || q != null) {
        int x = p == null ? 0 : p.val;
        int y = q == null ? 0 : q.val;
        int sum = x + y + carry;
        carry = sum / 10;
        curr.next = new ListNode(sum % 10);
        curr = curr.next;
        if (p != null) p = p.next;
        if (q != null) q = q.next;
    }
    if (carry > 0) {
        curr.next = new ListNode(carry);
    }
    return head.next;
}
```  

## 445. Add Two Numbers II

思路同上，使用两个stack来存储list的值, 后面相加的时候再pop就好了

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    Stack<Integer> s1 = new Stack<>();
    Stack<Integer> s2 = new Stack<>();
    while (l1 != null) {
        s1.push(l1.val);
        l1 = l1.next;
    }
    while (l2 != null) {
        s2.push(l2.val);
        l2 = l2.next;
    }
    int carry = 0;
    ListNode node = null;
    while (!s1.empty() || !s2.empty()) {
        int x = s1.empty() ? 0 : s1.pop();
        int y = s2.empty() ? 0 : s2.pop();
        int sum = x + y + carry;
        carry = sum / 10;
        ListNode n = new ListNode(sum % 10);
        n.next = node;
        node = n;
    }
    if (carry > 0) {
        ListNode head = new ListNode(carry);
        head.next = node;
        node = head;
    }
    return node;
}
```  

## 148. Sort List

Merge Sort, Recursion version, O(nlgn), O(n) space.

```java
class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        // use repeated doubling
        ListNode slow = head;
        ListNode fast = head;
        ListNode pre = head;
        while (fast != null && fast.next != null) { // in case of even or odd
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        pre.next = null; // 建立停止点，就是下一次递归时，sortList(head) 的停止点。
        return merge(sortList(head), sortList(slow));
    }

    public ListNode merge(ListNode slow, ListNode fast) {
        ListNode dummy = new ListNode(-1);
        ListNode curr = dummy;
        while (slow != null && fast != null) {
            if (slow.val < fast.val) {
                curr.next = slow;
                slow = slow.next;
            } else {
                curr.next = fast;
                fast = fast.next;
            }
            curr = curr.next;
        }
        if (slow != null) {
            curr.next = slow;
        }
        if (fast != null) {
            curr.next = fast;
        }
        return dummy.next;
    }
}
```  

## 23. Merge K Sorted List

1. Divide and Conquer, 比如合并6个链表，那么按照分治法，我们首先分别合并0和3，1和4，2和5.  这样下一次只需合并3个链表，我们再合并1和3，最后和2合并就可以了

```java
public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0) {
        return null;
    }
    int n = lists.length;
    while (n > 1) {
        int k = (n + 1) / 2;
        for (int i = 0; i < n / 2; i++) {
            lists[i] = mergeTwoLists(lists[i], lists[i + k]);
        }
        n = k;
    }
    return lists[0];
}

private ListNode mergeTwoLists(ListNode n1, ListNode n2) {
    ListNode head = new ListNode(-1);
    ListNode curr = head;
    while (n1 != null && n2 != null) {
        if (n1.val > n2.val) {
            curr.next = n2;
            n2 = n2.next;
        } else {
            curr.next = n1;
            n1 = n1.next;
        }
        curr = curr.next;
    }
    // 如果n1或者n2还有剩余的点
    if (n1 != null) {
        curr.next = n1;
    }
    if (n2 != null) {
        curr.next = n2;
    }
    return head.next;
}
```  

## 234. Palindrome Linked List

1. O(n) space, convert linked list to array, then solve it
2. O(1) space, convert second half of the list

```java
public boolean isPalindrome(ListNode head) {
    ListNode fast = head;
    ListNode slow = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    // odd nodes
    if (fast != null) {
        slow = slow.next;
    }

    slow = reverse(slow);
    fast = head;
    while (slow != null) {
        if (slow.val != fast.val) {
            return false;
        }
        slow = slow.next;
        fast = fast.next;
    }
    return true;
}

public ListNode reverse(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```  

### 141. Linked List Cycle

Two Pointer, 一快一慢

```java
public boolean hasCycle(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```  

## 138. Copy List With Random Pointer

O(n) Space, HashMap

```java
public Node copyRandomList(Node head) {
    Map<Node, Node> map = new HashMap<>();
    // store copy of each node
    Node node = head;
    while (node != null) {
        map.put(node, new Node(node.val));
        node = node.next;
    }
    // assign to new node
    node = head;
    while (node != null) {
        map.get(node).next = map.get(node.next);
        map.get(node).random = map.get(node.random);
        node = node.next;
    }
    return map.get(head);
}
```

O(1) space

```java
public Node copyRandomList(Node head) {
    if (head == null) return head;
    // 1 -> 2 -> 3-> 4
    Node pre = head;
    while (pre != null) { // 1->1->2->2->3->3->4->4
        Node clone = new Node(pre.val);
        clone.next = pre.next;
        pre.next = clone;
        pre = clone.next;
    }
    // Update random
    pre = head;
    while (pre != null) {
        pre.next.random = (pre.random == null) ? null : pre.random.next;
        pre = pre.next.next;
    }
    // seperate list
    pre = head;
    Node copyHead = head.next;
    Node copy = copyHead;
    while (copy != null) {
        pre.next = pre.next.next; // don't modify original list
        pre = pre.next;

        copy.next = (copy.next == null) ? null : copy.next.next;
        copy = copy.next;
    }
    return copyHead;
}
```

### 146. LRU Cache

思路，使用hashmap存储值，使用double linked list存储LRU

```java
class LRUCache {

    // Double Linkedlist Node inner class
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode pre;
        DLinkedNode next;
    }

    // Always add the new node right after head;
    private void addNode(DLinkedNode node) {
        node.next = head.next;
        node.pre = head;

        head.next.pre = node;
        head.next = node;
    }

    // Remove an existing node from the linked list.
    private void removeNode(DLinkedNode node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }

    // Move certain node in between to the head.
    private void moveToHead(DLinkedNode node) {
        this.removeNode(node);
        this.addNode(node);
    }

    // Pop the current tail
    private DLinkedNode popTail() {
        DLinkedNode res = tail.pre;
        this.removeNode(res);
        return res;
    }

    // Variables
    private Map<Integer, DLinkedNode> cache = new HashMap<>();
    private int count;
    private int capacity;
    private DLinkedNode head;
    private DLinkedNode tail;

    public LRUCache(int capacity) {
        this.count = 0;
        this.capacity = capacity;

        // Establish double linked list
        head = new DLinkedNode();
        tail = new DLinkedNode();

        head.pre = null;
        head.next = tail;

        tail.next = null;
        tail.pre = head;
    }

    public int get(int key) {
        DLinkedNode node = this.cache.get(key);
        if (node == null) { // Key doesn't exist
            return -1;
        }
        this.moveToHead(node); // update list
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = this.cache.get(key);

        if (node == null) { // key doesn't exist
            // add new node
            DLinkedNode newNode = new DLinkedNode();
            newNode.key = key;
            newNode.value = value;
            this.addNode(newNode);
            this.cache.put(key, newNode);
            count++;
            // if cache pass capacity
            if (count > capacity) {
                DLinkedNode tail = this.popTail();
                this.cache.remove(tail.key);
                count--;
            }
        } else { // key exist
            // update value
            node.value = value;
            this.moveToHead(node);
        }
    }
}
```  
