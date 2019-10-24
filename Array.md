# Questions

## 1. Two Sum

1. Brute Force, two level for loop.
2. One Pass Hash

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int n = target - nums[i];
        if (map.containsKey(n)) {
            return new int[] {i, map.get(n)};
        }
        map.put(nums[i], i);
    }
    return new int[] {};
}
```  

## 53. Maximum Subarray

1. 如果sum小于0， 不管下一个数是正是负，肯定会比这个数加上sum的值大

```java
public int maxSubArray(int[] nums) {
    int sum = Integer.MIN_VALUE;
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < nums.length; i++) {
        if (sum < 0) {
            sum = nums[i];
        } else {
            sum += nums[i];
        }
        max = Math.max(max, sum);
    }
    return max;
}
```  

## 121. Best time to buy and sell stock

1. 两个值，分别记录最大和最小，O(n)遍历一遍，如果price[i] < low更新low, 否则更新max

```java
public int maxProfit(int[] prices) {
    int profit = 0;
    if (prices.length < 2) {
        return profit;
    }
    int low = prices[0];
    for (int i = 0; i < prices.length; i++) {
        if (prices[i] < low) {
            low = prices[i];
        } else {
            profit = Math.max(profit, (prices[i] - low));
        }
    }
    return profit;
}
```  

## 88. Merge Sorted Array

1. 使用三个指针，m - 1, n - 1, m + n - 1，从右往左更新

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int i = m - 1;
    int j = n - 1;
    int k = m + n - 1;
    while (i >= 0 && j >= 0) {
        if (nums1[i] < nums2[j]) {
            nums1[k] = nums2[j];
            j--;
            k--;
        } else {
            nums1[k] = nums1[i];
            i--;
            k--;
        }
    }
    while (j >= 0) {
        nums1[k] = nums2[j];
        k--;
        j--;
    }
}
```  

## 283. Move Zeroes

1. 双指针，更新右指针，只有在右值不为0的时候交换左右值，同时更新左指针

```java
public void moveZeroes(int[] nums) {
    int l = 0;
    int r = 0;
    while (r < nums.length) {
        if (nums[r] != 0) {
            int tmp = nums[r];
            nums[r] = nums[l];
            nums[l] = tmp;
            l++;
        }
        r++;
    }
}
```  

## 167. Two Sum II - Input array is sorted

1. 双指针，一左一右

```java
public int[] twoSum(int[] numbers, int target) {
    if (numbers.length < 2) {
        return new int[2];
    }
    int left = 0;
    int right = numbers.length - 1;
    while (left < right) {
        int sum = numbers[left] + numbers[right];
        if (sum == target) {
            return new int[] {left + 1, right + 1};
        } else if (sum > target) {
            right--;
        } else {
            left++;
        }
    }
    return new int[2];
}
```  

## 170. Two Sum III - Data Structure Design

```java
class TwoSum {

    List<Integer> arr;
    /** Initialize your data structure here. */
    public TwoSum() {
        arr = new ArrayList<>();
    }
    /** Add the number to an internal data structure.. */
    public void add(int number) {
        arr.add(number);
    }
    /** Find if there exists any pair of numbers which sum is equal to the value. */
    public boolean find(int value) {
        /* 267 ms
        Set<Integer> set = new HashSet<>();
        for (int a : arr) {
            if (set.contains(a)) {
                return true;
            }
            set.add(value - a);
        }
        return false;
        */
        // 108 ms
        Collections.sort(arr, (a, b) -> a - b);
        int left = 0;
        int right = arr.size() - 1;
        while (left < right) {
            int sum = arr.get(left) + arr.get(right);
            if (sum == value) {
                return true;
            } else if (sum > value) {
                right--;
            } else {
                left++;
            }
        }
        return false;
    }
}
```  

## 268. Missing Number

1. 求和相减, 为了防止overflow, 减的操作提前执行。

```java
public int missingNumber(int[] nums) {
    int res = 0;
    for (int i = 0; i < nums.length; i++) {
        res += i - nums[i] + 1; // the number after the missing one will have zero
    }
    return res;
}
```  

## 26. Remove Duplicates from Sorted Array

1. Two Pointer, 如果右值不等于左值，左值右移一位，更新左值

```java
public int removeDuplicates(int[] nums) {
    int l = 0;
    for (int r = 1; r < nums.length; r++) {
        if (nums[r] != nums[l]) {
            l++;
            nums[l] = nums[r];
        }
    }
    return l + 1;
}
```  

## 977. Squares of Sorted Array

1. O(nLogn), 遍历数组，求平方，再sort
2. O(n), Two Pointer, 一左一右

```java
public int[] sortedSquares(int[] A) {
    int[] res = new int[A.length];
    if (A.length == 0) {
        return res;
    }

    int l = 0;
    int r = A.length - 1;
    int p = res.length - 1;
    while (l <= r) {
        int i = A[l] * A[l];
        int j = A[r] * A[r];
        if (i > j) {
            res[p] = i;
            p--;
            l++;
        } else {
            res[p] = j;
            p--;
            r--;
        }
    }
    return res;
}
```

## 15. Three Sum

1. Brute Force, three level for loop O(n^3)
2. Sort数组，第一个数One Pass，二三用双指针

```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums.length < 3) {
        return res;
    }
    Arrays.sort(nums);
    for (int i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) {
            continue;
        }
        int l = i + 1;
        int r = nums.length - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                l++;
                r--;
                while (l < r && nums[l] == nums[l - 1]) {
                    l++;
                }
            } else if (sum > 0) {
                r--;
            } else {
                l++;
            }
        }
    }
    return res;
}
```  

## 18. 4Sum

在3Sum的基础上，再套一层loop

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 4) {
            return res;
        }
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) { // skip duplicate
                continue;
            }
            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) { // skip duplicate
                    continue;
                }
                int left = j + 1;
                int right = n - 1;
                while (left < right) {
                    int sum = nums[left] + nums[right] + nums[i] + nums[j];
                    if (sum == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        left++;
                        right--;
                         // skip duplicate
                        while (left < right && nums[left] == nums[left - 1]) {
                            left++;
                        }
                         // skip duplicate
                        while (left < right && nums[right] == nums[right + 1]) {
                            right--;
                        }
                    } else if (sum > target) {
                        right--;
                    } else {
                        left++;
                    }
                }
            }
        }
        return res;
    }
}
```

## 454. 4Sum II

O(N^2), 将前两个数组的所有可能性算出来，用map存好 sum -> times

```java
class Solution {
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int n = A.length;
        int res = 0;
        if (n == 0) {
            return res;
        }
        // sum of A[] + B[] -> how many time it occurs
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = A[i] + B[j];
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0 - (C[i] + D[j]);
                if (map.containsKey(sum)) {
                    res += map.get(sum);
                }
            }
        }
        return res;
    }
}
```

## 56. Merge Intervals

1. Sort Interval的最小值, 使用interval[0]记录区域，判断下一个区域的起点和当前区域的终点，  
重合则更新当前区域终点，否则更新当前区域两端，使用comparator提高效率

```java
public int[][] merge(int[][] intervals) {
    if (intervals == null || intervals.length == 0) {
        return intervals;
    }
    List<int[]> res = new ArrayList<int[]>();
    Arrays.sort(intervals, new IntervalsComparator());
    int[] curInterval = intervals[0];
    res.add(curInterval);
    for (int[] interval : intervals) {
        if (curInterval[1] >= interval[0]) {
            curInterval[1] = Math.max(interval[1], curInterval[1]);
        } else {
            curInterval = interval;
            res.add(curInterval);
        }
    }
    return res.toArray(new int[res.size()][2]);
}

private class IntervalsComparator implements Comparator<int[]> {
    public int compare(int[] a, int[] b) {
        return a[0] - b[0];
    }
}
```

## 33. Search in Rotated Sorted Array

1. Binary Search, 判断中心点在哪个区间, 判断是否在线性的区间内

```java
public int search(int[] nums, int target) {
    int len = nums.length, left = 0, right = len - 1;
    if (left > right)
        return -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target)
            return mid;
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid])
                right = mid - 1;
            else
                left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }
    return -1;
}
```  

## 283. Product of Array Except Self

1. O(n) time, O(1) space, left -> right计算每个点左边的数的乘积,  
right -> left计算每个点右边数的乘积。

```java
public int[] productExceptSelf(int[] nums) {
    int[] res = new int[nums.length];
    res[0] = 1;
    for (int i = 1; i < nums.length; i++) {
        res[i] = nums[i - 1] * res[i - 1];
    }
    int r = 1;
    for (int i = nums.length - 1; i >= 0; i--) {
        res[i] = res[i] * r;
        r *= nums[i];
    }
    return res;
}
```  

## 11. Container With Most Water

1. 双指针, 一左一右, 两个指针的值比大小，更新指针

```java
public int maxArea(int[] height) {
    int l = 0;
    int r = height.length - 1;
    int max = 0;
    while (l < r) {
        max = Math.max(max, (r - l)*(Math.min(height[l], height[r])));
        if (height[l] > height[r]) {
            r--;
        } else {
            l++;
        }
    }
    return max;
}
```  

## 289. Game of Life

1. O(1) space and O(mn) time, copy数组，根据copy计算每个点新值，更新原始board

```java
public void gameOfLife(int[][] board) {

    int[] neighbors = {0, 1, -1};
    int rows = board.length;
    int cols = board[0].length;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int liveNeighbors = 0;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {

                    if (!(neighbors[i] == 0 && neighbors[j] == 0)) {
                        int r = (row + neighbors[i]);
                        int c = (col + neighbors[j]);
                        if ((r < rows && r >= 0) && (c < cols && c >= 0) && (Math.abs(board[r][c]) == 1)) {
                            liveNeighbors += 1;
                        }
                    }
                }
            }

            if ((board[row][col] == 1) && (liveNeighbors < 2 || liveNeighbors > 3)) {
                board[row][col] = -1; // Was live, now dead
            }
            if (board[row][col] == 0 && liveNeighbors == 3) {
                board[row][col] = 2; // Was dead, now live
            }
        }
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            if (board[row][col] > 0) {
                board[row][col] = 1;
            } else {
                board[row][col] = 0;
            }
        }
    }
}
```  

## 31. Next Permutation

1. 找到下一个全排列，首先找到第一个下降点，从下降序列中找到刚好大于下降点的点，交换两个点，将后面的序列reverse

```java
public void nextPermutation(int[] nums) {
    int i = nums.length - 2;
    while (i >= 0 && nums[i + 1] <= nums[i]) { // find the first decreasing index
        i--;
    }

    if (i >= 0) {
        int j = nums.length - 1;
        while (j >= 0 && nums[j] <= nums[i]) { // find the index just greater than nums[i]
            j--;
        }
        swap(nums, i, j);
    }
    reverse(nums, i + 1);
}

private void swap(int[] nums, int i, int j) {
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}

private void reverse(int[] nums, int start) {
    int i = start;
    int j = nums.length - 1;
    while (i < j) {
        swap(nums, i, j);
        i++;
        j--;
    }
}
```  

## 54. Spiral Matrix

1. Layer by Layer
![define layer](54_spiralmatrix.png)

```java
public List<Integer> spiralOrder(int[][] matrix) {
    List<Integer> list = new ArrayList<>();
    if (matrix.length == 0) {
        return list;
    }

    int r1 = 0;
    int r2 = matrix.length - 1;
    int c1 = 0;
    int c2 = matrix[0].length - 1;
    while (r1 <= r2 && c1 <= c2) {
        for (int c = c1; c <= c2; c++) {
            list.add(matrix[r1][c]);
        }
        for (int r = r1 + 1; r <= r2; r++) {
            list.add(matrix[r][c2]);
        }
        if (r1 < r2 && c1 < c2) {
            for (int c = c2 - 1; c > c1; c--) {
                list.add(matrix[r2][c]);
            }
            for (int r = r2; r > r1; r--) {
                list.add(matrix[r][c1]);
            }
        }
        r1++;
        r2--;
        c1++;
        c2--;
    }
    return list;
}
```  

## 34. Find First and Last Position of Element in Sorted Array

分两步，首先找到first, 然后找到last

```java
public int[] searchRange(int[] nums, int target) {

    int[] res = {-1, -1};
    if (nums == null || nums.length == 0) {
        return res;
    }

    int lo = 0;
    int hi = nums.length - 1;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (target <= nums[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if (nums[lo] == target) {
        res[0] = lo;
    } else if (nums[hi] == target) {
        res[0] = hi;
    }

    lo = 0;
    hi = nums.length - 1;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (target >= nums[mid]) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (nums[hi] == target) {
        res[1] = hi;
    } else if (nums[lo] == target) {
        res[1] = lo;
    }

    return res;
}
```  

## 4. Median of Two Sorted Array

1. 找到中位数，由于m+n奇偶性不确定，trick: 找到(m+n+1)/2和(m+n+2)/2取平均值  
[解释](https://blog.csdn.net/hk2291976/article/details/51107778)

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int n = nums1.length;
    int m = nums2.length;
    if (n > m) {
        return findMedianSortedArrays(nums2, nums1); // make sure n is smaller
    }

    int l1, l2, r1, r2, c1, c2, lo;
    l1 = l2 = r1 = r2 = c1 = c2 = lo = 0;
    int hi = 2 * n;
    while (lo <= hi) {
        c1 = (lo + hi) / 2;
        c2 = m + n - c1;
        l1 = (c1 == 0) ? Integer.MIN_VALUE : nums1[(c1 - 1) / 2];
        r1 = (c1 == 2 * n) ? Integer.MAX_VALUE : nums1[c1 / 2];
        l2 = (c2 == 0) ? Integer.MIN_VALUE : nums2[(c2 - 1) / 2];
        r2 = (c2 == 2 * m) ? Integer.MAX_VALUE : nums2[c2 / 2];

        if (l1 > r2) {
            hi = c1 - 1;
        } else if (l2 > r1) {
            lo = c1 + 1;
        } else {
            break;
        }
    }
    return (Math.max(l1, l2) + Math.min(r1, r2)) / 2.0;
}
```  

## 42. Trapping Rain Water

1. Two Pointer.

```java
public int trap(int[] height) {
    if (height.length < 3) {
        return 0;
    }

    int vol = 0;
    int l = 0;
    int r = height.length - 1;
    while (l < r && height[l] <= height[l + 1]) l++;
    while (l < r && height[r] <= height[r - 1]) r--;

    while (l < r) {
        int left = height[l];
        int right = height[r];

        if (left <= right) {
            while (l < r && left >= height[++l]) {
                vol += left - height[l];
            }
        } else {
            while (l < r && right >= height[--r]) {
                vol += right - height[r];
            }
        }
    }
    return vol;
}
```  

## 986. Interval List Intersections

双指针，在两个数组从左向右遍历

```java
public int[][] intervalIntersection(int[][] A, int[][] B) {
    List<int[]> list = new ArrayList<>();
    int m = A.length;
    int n = B.length;
    int i = 0, j = 0;
    int startMax = Integer.MIN_VALUE;
    int endMin = Integer.MAX_VALUE;
    while (i < m && j < n) {
        startMax = Math.max(A[i][0], B[j][0]);
        endMin = Math.min(A[i][1], B[j][1]);
        if (endMin >= startMax) { // 比较是否有重合部分
            list.add(new int[]{startMax, endMin});
        }
        if (A[i][1] == endMin) {
            i++;
        }
        if (B[j][1] == endMin){
            j++;
        }
    }
    return list.toArray(new int[list.size()][2]);
}
```  

## 528. Random Pick With Weight

比如若权重数组为 [1, 3, 2] 的话，那么累加和数组为 [1, 4, 6]，整个的权重和为6，我们 rand() % 6，可以随机出范围 [0, 5] 内的数,  
随机到 0 则为第一个点，随机到 1，2，3 则为第二个点，随机到 4，5则为第三个点,  所以我们随机出一个数字x后，然后再累加和数组中查找第一个大于随机数x的数字，使用二分查找法可以找到第一个大于随机数x的数字的坐标，即为所求

```java
int[] sum;
Random random;

public Solution(int[] w) {
    this.random = new Random();
    for (int i = 1; i < w.length; i++) {
        w[i] += w[i - 1];
    }
    this.sum = w;
}
public int pickIndex() {
    int pick = random.nextInt(sum[sum.length - 1]) + 1;
    int left = 0;
    int right = sum.length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (pick == sum[mid]) {
            return mid;
        } else if (pick > sum[mid]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}
```  

## 1231. Divide Chocolate

Binary Search, left, mid, right表示可以切成的长度，找到符合范围的这个长度的最大值  
这个长度同时也是所有切好的块数中，长度最短的  
限制条件，切成的块数一定要大于K，也就是要给k+1个人吃

```java
class Solution {
    public int maximizeSweetness(int[] sweetness, int K) {
        int left = 1;
        int right = 0;
        for (int num : sweetness) {
            right += num;
        }
        while (left < right) {
            int mid = (left + right + 1) / 2; // 每次都拿到长度的偏大值
            int cur = 0;
            int cuts = 0;
            for (int num : sweetness) {
                cur += num;
                if (cur >= mid) { // 甜度和达到要求
                    cur = 0; // 重新计算甜度和
                    cuts++;
                    if (cuts > K) { // 分的块数符合要求
                        break;
                    }
                }
            }
            if (cuts > K) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
}
```  
