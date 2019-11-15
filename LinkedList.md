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

## 92. Reverse Linked List II

```java
// reverse order
// [1, 2, 3, 4, 5], 2, 4
// [1, 3, 2, 4, 5] -> [1, 4, 3, 2, 5]
class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy;
        pre.next = head;
        for (int i = 0; i < m - 1; i++) {
            pre = pre.next;
        }
        ListNode cur = pre.next;
        for (int i = m; i < n; i++) {
            ListNode node = cur.next;
            cur.next = node.next;
            node.next = pre.next;
            pre.next = node;
        }
        return dummy.next;
    }
}
```  

## 83. Remove Duplicates from Sorted List

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if (cur.next.val == cur.val) {
                cur.next = cur.next.next;  
            } else {
                cur = cur.next;
            }
        }
        return head;
    }
}
```  

## 82. Remove Duplicates from Sorted List II

```java
// iteration
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy;
        ListNode cur = head;
        pre.next = cur;
        while (cur != null) {
            while (cur.next != null && cur.next.val == cur.val) {
                cur = cur.next; // 到达最后一个重复的node
            }
            if (pre.next != cur) {
                pre.next = cur.next;
                cur = cur.next;
            } else {
                pre = pre.next;
                cur = cur.next;
            }
        }
        return dummy.next;
    }
}
// recursion
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next != null && head.next.val == head.val) {
            while (head.next != null && head.next.val == head.val) {
                head = head.next;
            }
            return deleteDuplicates(head.next);
        } else {
            head.next = deleteDuplicates(head.next);
        }
        return head;
    }
}
```  

## 203. Remove Linked List Elements

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        ListNode pre = dummy;
        ListNode cur = head;
        pre.next = cur;
        while (cur != null) {
            if (cur.val != val) {
                pre = pre.next;
                cur = cur.next;
            } else {
                pre.next = cur.next;
                cur = pre.next;
            }
        }
        return dummy.next;
    }
}
```  

## 876. Middle of the Linked List

```java
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
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

```java
// 定义carry记录进位
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

## 143. Reorder List

```java
// 这道链表重排序问题可以拆分为以下三个小问题：
// 1. 使用快慢指针来找到链表的中点，并将链表从中点处断开，形成两个独立的链表。
// 2. 将第二个链翻转。
// 3. 将第二个链表的元素间隔地插入第一个链表中。
class Solution {
    public void reorderList(ListNode head) {
        if (head == null) {
            return;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        } // slow到达了 1,2,3,4的2，1,2,3,4,5的3
        ListNode second = slow.next;
        slow.next = null;
        ListNode pre = null;
        while (second != null) {
            ListNode node = second.next;
            second.next = pre;
            pre = second;
            second = node;
        }
        ListNode first = head;
        ListNode dummy = first;
        while (first != null && pre != null) {
            ListNode tmp = first.next;
            first.next = pre;
            pre = pre.next;
            first.next.next = tmp;
            first = tmp;
        }
    }
}
```  

## 430. Flatten a Multilevel Doubly Linked List

```java
// 思路：每到一个子节点不为空的节点时，将该节点的next存入stack，当该节点下一层的节点遍历完后
// 再将next节点pop出来
class Solution {
    public Node flatten(Node head) {
        Stack<Node> s = new Stack<>();
        Node cur = head;
        while (cur != null) {
            if (cur.child != null) {
                s.push(cur.next);
                cur.next = cur.child;
                cur.child = null;
                if (cur.next != null) {
                    cur.next.prev = cur;
                }
            } else if (cur.next == null && !s.empty()) {
                cur.next = s.pop();
                if (cur.next != null) {
                    cur.next.prev = cur;
                }
            }
            cur = cur.next;
        }
        return head;
    }
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

## 141. Linked List Cycle

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

## 142. Linked List Cycle II

```java
// O(N) time complexity
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                break;
            }
        }
        ListNode find = head;
        while (slow != find) {
            slow = slow.next;
            find = find.next;
        }
        return find;
    }
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

## 146. LRU Cache

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

## 25. Reverse Nodes in k-Group

```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        while (prev != null) {
            prev = reverse(prev, k);
        }
        return dummy.next;
    }

    public ListNode reverse(ListNode prev, int k) {
        ListNode last = prev;
        for (int i = 0; i <= k; i++) {
            last = last.next;
            // i != k to prevent [1, 2, 3], k = 3, not enough element to reverse
            if (i != k && last == null) return null;
        }
        ListNode tail = prev.next; // tail become the last element after reverse
        ListNode cur = prev.next.next;
        while (cur != last) {
            ListNode next = cur.next;
            cur.next = prev.next;
            prev.next = cur;
            tail.next = next;
            cur = next;
        }
        return tail;
    }
}
```  
