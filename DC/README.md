### 973. K Closet Point to Origin
思路：使用一个数组，记录每个点的距离，将这个数组排序，我们需要K个点，所以距离为这个数组的前K位的点就是我们的目标
```java
public int[][] kClosest(int[][] points, int K) {
    int[] power = new int[points.length];
    for (int i = 0; i < points.length; i++) {
        power[i] = cal(points[i][0], points[i][1]);
    }

    Arrays.sort(power);
    int target = power[K - 1];
    int[][] res = new int[K][2];
    int index = 0;
    for (int[] point : points) {
        if (cal(point[0], point[1]) <= target) {
            res[index] = point;
            index++;
        }
    }
    return res;
}

public int cal(int i, int j) {
    return i * i + j * j;
}
```  
2. PriorityQueue, 保持queue的size为k，在放入queue的时候，降序放入
```java
public int[][] kClosest(int[][] points, int K) {
    PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparing(a -> -a[0] * a[0] - a[1] * a[1]));
    for (int[] p : points) {
        pq.offer(p);
        if (pq.size() > K) { // poll out the farthest among the K + 1 points.
            pq.poll();
        }
    }
    int[][] ans = new int[K][2];
    while (K > 0) {
        K--;
        ans[K] = pq.poll();
    }
    return ans;
}
```  

### 215. Kth Largest Element in Array
1. Sort
2. PriorityQueue, add 默认为升序，判断queue的size和值的大小
```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    for (int num : nums) {
        if (pq.size() < k || num > pq.peek()) {
            pq.add(num);
        }
        if (pq.size() > k) {
            pq.remove();
        }
    }
    return pq.peek();
}
```  
