## Easy
### 206. Reversed LinkedList
```java
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

### 21. Merge Two Sorted Lists
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
2. Iterator
```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode preHead = new ListNode(-1);
    ListNode pre = preHead;
    
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            pre.next = l1;
            l1 = l1.next;
        } else {
            pre.next = l2;
            l2 = l2.next;
        }
        pre = pre.next;
    }
    if (l1 == null) {
        pre.next = l2;
    }
    if (l2 == null) {
        pre.next = l1;
    }
    return preHead.next;
}
```  

### 2. Add Two Numbers
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
