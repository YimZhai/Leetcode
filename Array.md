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

## 525. Contiguous Array

```java
// 使用一个HashMap存储累价值和对应的下标
// 在遍历的过程中，遇到0，-1，遇到1，+1
// 时间复杂度：O(N)，空间O(N)
class Solution {
    public int findMaxLength(int[] nums) {
        int sum = 0;
        int res = 0;
        // sum -> index
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < nums.length; i++) {
            sum += (nums[i] == 0) ? -1 : 1;
            if (map.containsKey(sum)) {
                res = Math.max(res, i - map.get(sum));
            } else {
                map.put(sum, i);
            }
        }
        return res;
    }
}
```  

## 53. Maximum Subarray

```java
// 如果sum小于0， 不管下一个数是正是负，肯定会比这个数加上sum的值大
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

## 836. Rectangle Overlap

```java
class Solution {
    public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
        if (rec1[2] > rec2[0] && rec1[0] < rec2[2] &&
            rec1[3] > rec2[1] && rec1[1] < rec2[3]) {
            return true;
        }
        return false;
    }
}
```  

## 152. Maximum Product Subarray

```java
// 用两个数组，分别存当前[0, i]的子数组，包括nums[i]时的最大值和最小值,
// 每到i的位置时，当前位置的最大值和最小值肯定来自于max[i-1]*nums[i], min[i-1]*nums[i], nums[i]其中之一
// O(N)时间和空间
class Solution {
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[] m = new int[n];
        int[] s = new int[n];
        int res= nums[0];
        m[0] = nums[0];
        s[0] = nums[0];
        for (int i = 1; i < n; i++) {
            m[i] = Math.max(Math.max(m[i - 1] * nums[i], s[i - 1] * nums[i]), nums[i]);
            s[i] = Math.min(Math.min(m[i - 1] * nums[i], s[i - 1] * nums[i]), nums[i]);
            res = Math.max(res, m[i]);
        }
        return res;
    }
}
```  

## 561. Array Partition I

```java
class Solution {
    public int arrayPairSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int sum = 0;
        for (int i = 0; i < n - 1; i += 2) {
            sum += nums[i];
        }
        return sum;
    }
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

## 912. Sort an Array

Array排序用quick sort, LinkedList排序用merge sort

1. Merge Sort

```java
class Solution {
    public int[] sortArray(int[] nums) {
        mSort(nums, 0, nums.length - 1);
        return nums;
    }

    public void mSort(int[] nums, int left, int right) {
        if (left >= right) {
            return;
        }

        int mid = (left + right) / 2;
        mSort(nums, left, mid);
        mSort(nums, mid + 1, right);
        merge(nums, left, mid, right);
    }

    public void merge(int[] nums, int left, int mid, int right) {
        int i = left;
        int j = mid + 1;
        int[] tmp = new int[right - left + 1];
        int index = 0;
        while (i <= mid && j <= right) {
            if (nums[i] < nums[j]) {
                tmp[index++] = nums[i++];
            } else {
                tmp[index++] = nums[j++];
            }
        }
        while (i <= mid) {
            tmp[index++] = nums[i++];
        }
        while (j <= right) {
            tmp[index++] = nums[j++];
        }
        for (int k = 0; k < tmp.length; k++) {
            nums[left + k] = tmp[k];
        }
    }
}
```  

2.Quick Sort

```java
class Solution {
    public int[] sortArray(int[] nums) {
        qSort(nums, 0, nums.length - 1);
        return nums;
    }

    public void qSort(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }

        // 选择pivot点，nums[start], nums[mid], nums[random]
        int pivot = nums[start];
        int left = start;
        int right = end;
        while (left <= right) {
            while (left <= right && nums[left] < pivot) {
                left++;
            }
            while (left <= right && nums[right] > pivot) {
                right--;
            }

            if (left <= right) {
                int tmp = nums[left];
                nums[left] = nums[right];
                nums[right] = tmp;
                left++;
                right--;
            }
        }
        qSort(nums, start, right);
        qSort(nums, left, end);
    }
}
```  

## 283. Move Zeroes

```java
// 1. 双指针，更新右指针，只有在右值不为0的时候交换左右值，同时更新左指针
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

```java
// 1. 双指针，一左一右
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

## 27. Remove Element

```java
// O(N)
class Solution {
    public int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != val) {
                nums[i] = nums[j]; // 直接更新，不需要考虑i之后的元素
                i++;
            }
        }
        return i;
    }
}
```  

## 485. Max Consecutive Ones

```java
// O(N)
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int res = 0;
        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                cnt++;
                res = Math.max(res, cnt);
            } else {
                cnt = 0;
            }
        }
        return res;
    }
}
```  

## 209. Minimum Size Subarray Sum

```java
// 双指针，快的指针累加每个值，超过目标值后，慢指针更新，同时更新结果
// O(N)
class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        int j = 0;
        int sum = 0;
        int i = 0;
        int res = Integer.MAX_VALUE;
        while (j < nums.length) {
            sum += nums[j];
            j++;
            while (sum >= s) {
                res = Math.min(res, j - i);
                sum -= nums[i];
                i++;
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }
}
```  

## 724. Find Pivot Index

```java
class Solution {
    public int pivotIndex(int[] nums) {
        if (nums.length < 3) {
            return -1;
        }
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        int left = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > 0) {
                left += nums[i - 1];
            }
            int right = sum - nums[i] - left;
            if (left == right) {
                return i;
            }
        }
        return -1;
    }
}
```  

## 747. Largest Number At Least Twice of Others

```java
class Solution {
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) {
            return 0;
        }
        int res = 0;
        int first = Integer.MIN_VALUE;
        int second = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > first) {
                second = first;
                first = nums[i];
                res = i;
            } else if (nums[i] > second) {
                second = nums[i];
            }
        }
        return first >= second * 2 ? res : -1;
    }
}
```  

## 66. Plus One

```java
class Solution {
    public int[] plusOne(int[] digits) {
        int len = digits.length;
        for (int i = len - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            }
            digits[i] = 0;
        }
        // 处理999...的情况
        int[] res = new int[len + 1];
        res[0] = 1;
        return res;
    }
}
```  

## 374 Guess Number Higher or Lower

```java
public class Solution extends GuessGame {
    public int guessNumber(int n) {
        int lo = 1;
        int hi = n;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            int val = guess(mid);
            if (val == 0) {
                return mid;
            } else if (val < 0) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return -1;
    }
}
```  

## 375 Guess Number Higher or Lower II

## 498. Diagonal Traverse

```java
class Solution {
    public int[] findDiagonalOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new int[0];
        }
        int row = matrix.length;
        int col = matrix[0].length;
        int[] res = new int[row*col];
        int r = 0;
        int c = 0;
        int d = 1; // 初始化向上走
        for (int i = 0; i < res.length; i++) {
            res[i] = matrix[r][c];
            r -= d;
            c += d;
            // 当遇到右上或者左下角的时候，会同时满足其中两个if条件
            // 如果先判断先判断是否 < 0, 会导致两个if条件同时进入
            // 此时方向调转了两次，会出现越界的情况
            if (r >= row) {
                r = row - 1;
                c += 2;
                d = -d;
            }
            if (c >= col) {
                c = col - 1;
                r += 2;
                d = -d;
            }
            if (r < 0) {
                r = 0;
                d = -d;
            }
            if (c < 0) {
                c = 0;
                d = -d;
            }
        }
        return res;
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

## 41. First Missing Integer

O(n) time and space 方法，很直观，用一个set存储出现过的值, 同时记录最大值, 然后从1开始遍历

```java
public int firstMissingPositive(int[] nums) {
    int max = 0;
    Set<Integer> set = new HashSet<>();
    for (int num : nums) {
        max = Math.max(max, num);
        set.add(num);
    }
    for (int i = 1; i <= max; i++) {
        if (!set.contains(i)) {
            return i;
        }
    }
    return max + 1;
}
```  

O(n) time and O(1) space

```java
public int firstMissingPositive(int[] nums) {
    int n = nums.length;
    for (int i = 0; i < n; i++) {
        // nums[i] 需要在数组的范围内, 同时nums[i]不在应该在的位置上
        while (nums[i] > 0 && nums[i] < n && nums[i] != nums[nums[i] - 1]) {
            int tmp = nums[i];
            nums[i] = nums[nums[i] - 1];
            nums[tmp - 1] = tmp;
        }
    }
    for (int i = 0; i < n; i++) {
        // 如果该值和下标不对应则返回
        if (nums[i] != i + 1) {
            return i + 1;
        }
    }
    return n + 1;
}
```  

## 26. Remove Duplicates from Sorted Array

```java
// Two Pointer, 如果右值不等于左值，左值右移一位，更新左值
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

## 240. Search a 2D Matrix II

```java
// O(MN) TLE
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int n = matrix.length;
        if (n == 0) {
            return false;
        }
        int m = matrix[0].length;
        return helper(matrix, 0, 0, target);
    }

    public boolean helper(int[][] matrix, int r, int c, int target) {
        if (r == matrix.length || c == matrix[0].length) {
            return false;
        }
        if (target < matrix[r][c]) {
            return false;
        }
        if (target == matrix[r][c]) {
            return true;
        }
        return helper(matrix, r + 1, c, target) || helper(matrix, r, c + 1, target);
    }
}
```

```java
// O(M+N) solution, 从左下角或者右上角开始遍历
// 可以将该图理解为两个root的BST
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int n = matrix.length;
        if (n == 0) {
            return false;
        }
        int m = matrix[0].length;
        int i = n - 1;
        int j = 0;
        while (i >= 0 && j < m) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                i--;
            } else {
                j++;
            }
        }
        return false;
    }
}
```  

## 48 Rotate Image

```java
// 先沿右上对角线反转，再沿着水平中线反转
// time complexity: O(N^2)
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][n - 1 - i];
                matrix[n - 1 - j][n - 1 - i] = tmp;
            }
        }
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = tmp;
            }
        }
    }
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

## 704. Binary Search

```java
class Solution {
    public int search(int[] nums, int target) {
        int lo = 0;
        int hi = nums.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return -1;
    }
}
```  

## 33. Search in Rotated Sorted Array

```java
// Binary Search, 判断中心点在哪个区间, 判断是否在线性的区间内
class Solution {
    public int search(int[] nums, int target) {
        int lo = 0;
        int hi = nums.length - 1;
        while (lo <= hi) {
            int mid = (hi + lo) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[lo] <= nums[mid]) {
                if (nums[lo] <= target && target < nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }
}
```  

### 189. Rotate Array

```java
extra space，目标数组的下标通过取余的方式算
public void rotate(int[] nums, int k) {
    int len = nums.length;
    int[] res = new int[len];
    for (int i = 0; i < len; i++) {
        res[(i + k) % len] = nums[i];
    }
    for (int i = 0; i < len; i++) {
        nums[i] = res[i];
    }
}
```  

O(1) space solution

```java
lass Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    private void reverse(int[] nums, int left, int right) {
        while (left < right) {
            int tmp = nums[left];
            nums[left] = nums[right];
            nums[right] = tmp;
            left++;
            right--;
        }
    }
}
```  

## 278. First Bad Version

```java
public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        int lo = 1;
        int hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (isBadVersion(mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if (isBadVersion(lo)) {
            return lo;
        } else {
            return hi;
        }
    }
}
```  

## 162. Find Peak Element

```java
class Solution {
    public int findPeakElement(int[] nums) {
        int lo = 0;
        int hi = nums.length - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] > nums[mid + 1]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
}
```  

## 153. Find Minimum in Rotated Sorted Array

```java
class Solution {
    public int findMin(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int lo = 0;
        int hi = nums.length - 1;
        if (nums[hi] > nums[lo]) {
            return nums[lo];
        }
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            System.out.println(mid);
            if (nums[mid] > nums[mid + 1]) {
                return nums[mid + 1];
            }
            if (nums[mid - 1] > nums[mid]) {
                return nums[mid];
            }
            if (nums[mid] > nums[0]) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }
}
```  

## 154. Find Minimum in Rotated Sorted Array II

## 1007. Minimum Domino Rotations For Equal Row

```java
public int minDominoRotations(int[] A, int[] B) {
    int len = A.length;
    int[] cntA = new int[7];
    int[] cntB = new int[7];
    int[] same = new int[7];
    for (int i = 0; i < len; i++) {
        cntA[A[i]]++;
        cntB[B[i]]++;
        if (A[i] == B[i]) {
            same[A[i]]++;
        }
    }
    for (int i = 0; i < 7; i++) {
        if (cntA[i] + cntB[i] - same[i] >= len) {
            return Math.min(cntA[i], cntB[i]) - same[i];
        }
    }
    return -1;
}
```

## 238. Product of Array Except Self

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

## 315. Count of Smaller Numbers After Itself

1. Brute Force, O(n^2), 每个点遍历一遍后面所有节点，统计个数

```java
public List<Integer> countSmaller(int[] nums) {
    if (nums.length == 0) {
        return new ArrayList<Integer>();
    }
    Integer[] cnts = new Integer[nums.length];
    for (int i = 0; i < cnts.length; i++) {
        int cnt = 0;
        for (int j = i + 1; j < cnts.length; j++) {
            if (nums[j] < nums[i]) {
                cnt++;
            }
        }
        cnts[i] = cnt;
    }
    cnts[cnts.length - 1] = 0;
    return Arrays.asList(cnts);
}
```  

1. Merge Sort, O(nlgn), 保留原数组，根据原数组的大小sort他们的index

```java
class Solution {
    int[] count;
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int[] indexes = new int[nums.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        count = new int[nums.length];
        mergeSort(nums, indexes, 0, nums.length - 1);
        for (int i = 0; i < count.length; i++) {
            res.add(count[i]);
        }
        return res;
    }

    public void mergeSort(int[] nums, int[] indexes, int left, int right) {
        if (left >= right) {
            return;
        }

        int mid = (left + right) / 2;
        mergeSort(nums, indexes, left, mid);
        mergeSort(nums, indexes, mid + 1, right);
        merge(nums, indexes, left, mid, right);
    }

    public void merge(int[] nums, int[] indexes, int left, int mid, int right) {
        int[] tmp = new int[right - left + 1];
        int i = left;
        int j = mid + 1;
        int k = 0;
        // 右半sort部分小于左半部分的数字的count
        int rightCnt = 0;
        while (i <= mid && j <= right) {
            // indexes排序，根据nums[index]的大小
            if (nums[indexes[i]] > nums[indexes[j]]) { // 如果右边小,右边的cnt++
                tmp[k] = indexes[j];
                rightCnt++;
                j++;
            } else { // 如果左边小右边大，更新左边的count值
                tmp[k] = indexes[i];
                count[indexes[i]] += rightCnt;
                i++;
            }
            k++;
        }
        while (i <= mid) {
            tmp[k] = indexes[i];
            count[indexes[i]] += rightCnt;
            i++;
            k++;
        }
        while (j <= right) {
            tmp[k++] = indexes[j++];
        }
        for (int p = 0; p < tmp.length; p++) {
            indexes[p + left] = tmp[p];
        }
    }
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

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix.length == 0) {
            return res;
        }
        int r1 = 0;
        int c1 = 0;
        int r2 = matrix.length - 1;
        int c2 = matrix[0].length - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) {
                res.add(matrix[r1][c]);
            }
            r1++;
            for (int r = r1; r <= r2; r++) {
                res.add(matrix[r][c2]);
            }
            c2--;
            if (r1 > r2 || c1 > c2) {
                break;
            }
            for (int c = c2; c >= c1; c--) {
                res.add(matrix[r2][c]);
            }
            r2--;
            for (int r = r2; r >= r1; r--) {
                res.add(matrix[r][c1]);
            }
            c1++;
        }
        return res;
    }
}
```  

## 59. Spiral Matrix II

```java
// 思路同上
class Solution {
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int r1 = 0;
        int r2 = n - 1;
        int c1 = 0;
        int c2 = n - 1;
        int num = 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) {
                res[r1][c] = num++;
            }
            for (int r = r1 + 1; r <= r2; r++) {
                res[r][c2] = num++;
            }
            if (r1 < r2 && c1 < c2) {
                for (int c = c2 - 1; c > c1; c--) {
                    res[r2][c] = num++;
                }
                for (int r = r2; r > r1; r--) {
                    res[r][c1] = num++;
                }
            }
            r1++;
            r2--;
            c1++;
            c2--;
        }
        return res;
    }
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

## 55. Jump Game

```java
// Greedy
class Solution {
    public boolean canJump(int[] nums) {
        int max = 0; // 当前可以走到最大下标
        for (int i = 0; i < nums.length; i++) {
            if (i > max) {
                return false;
            }
            max = Math.max(max, nums[i] + i);
        }
        return true;
    }
}
```  

## 45. Jump Game II

```java
class Solution {
    public int jump(int[] nums) {
        // index level
        // 2
        // 3, 1
        // 1, 4
        // ...
        if (nums.length < 2) {
            return 0;
        }
        int curMax = 0;
        int step = 0;
        int i = 0;
        while (i <= curMax) {
            step++;
            // rightMost, 每一层可以到达的最远距离
            int rightMost = curMax;
            for (; i <= curMax; i++) {
                rightMost = Math.max(rightMost, nums[i] + i);
                if (rightMost >= nums.length - 1) {
                    return step;
                }
            }
            curMax = rightMost;
        }
        return -1;
    }
}
```  

## 735. Asteroid Collision

```java
// 新建一个list，开始遍历原始数组，从左到右，对遇到元素进行判断
// 是否大于0，和之前list最后的元素数值的比较
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        LinkedList<Integer> list = new LinkedList<>();
        for (int a : asteroids) {
            if (a > 0) {
                list.add(a);
            } else {
                // 新进入的行星向左移动，摧毁所有比他小的行星
                while (!list.isEmpty() && list.peekLast() > 0 && list.peekLast() + a < 0) {
                    list.pollLast();
                }
                // 两个行星大小一样
                if (!list.isEmpty() && list.peekLast() + a == 0) {
                    list.pollLast();
                } else if (list.isEmpty() || list.peekLast() < 0) {
                    // 与之前的行星不会发生碰撞
                    list.add(a);
                }
            }
        }
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }
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

## 957. Prison Cells After N Days

```java
class Solution {
    public int[] prisonAfterNDays(int[] cells, int N) {
        int[] first = new int[8];
        for (int i = 1; i < 7; i++) {
            if (cells[i - 1] == cells[i + 1]) {
                first[i] = 1;
            } else {
                first[i] = 0;
            }
        }
        cells = first.clone();
        N--;
        int cycle = 1;
        while (N > 0) {
            N--;
            int[] next = new int[8];
            for (int i = 1; i < 7; i++) {
                if (cells[i - 1] == cells[i + 1]) {
                    next[i] = 1;
                } else {
                    next[i] = 0;
                }
            }
            if (Arrays.equals(next, first)) {
                N %= cycle;
            }
            cells = next.clone();
            cycle++;
        }
        return cells;
    }
}
```  

## 84. Largest Rectangle in Histogram

```java
// 直观的思路，两个指针从左向右扫，O(N^2)时间
// 优化，使用stack，时间降低到 O(N)
class Solution {
   public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        int maxArea = 0;
        // 建立stack只存储比当前stack中最大高度大的bar的index
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i <= len;) {
            int height = (i == len) ? 0 : heights[i];
            // 将0加入stack应对[1]的情况
            // 只有在新的高度入栈的时候才向右移动游标
            if (stack.empty() || height > heights[stack.peek()]) {
                stack.push(i);
                i++;
            } else {
                // 此时i到了目前最大高度的右边第一个小于左边的值
                int curMaxHeight = heights[stack.pop()]; // 获取当前的最大高度
                int right = i - 1; // 右边界
                int left = stack.empty() ? 0 : stack.peek() + 1; // 左边界
                int width = right - left + 1;
                maxArea = Math.max(maxArea, curMaxHeight * width);
            }
        }
        return maxArea;
    }
}
```  

## 85. Maximal Rectangle

```java
// 思路同上，分层遍历二维数组，使用一个数组，记录每个col的高度
// 每遍历完一层，调用上一题的函数，更新返回结果
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int res = 0;
        int[] height = new int[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '0') {
                    height[j] = 0;
                } else {
                    height[j]++;
                }
            }
            res = Math.max(res, largestRectangleArea(height));
        }
        return res;
    }

    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        int maxArea = 0;
        // 建立stack只存储比当前stack中最大高度大的bar的index
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i <= len;) {
            int height = (i == len) ? 0 : heights[i];
            // 将0加入stack应对[1]的情况
            // 只有在新的高度入栈的时候才向右移动游标
            if (stack.empty() || height > heights[stack.peek()]) {
                stack.push(i);
                i++;
            } else {
                // 此时i到了目前最大高度的右边第一个小于左边的值
                int curMaxHeight = heights[stack.pop()]; // 获取当前的最大高度
                int right = i - 1; // 右边界
                int left = stack.empty() ? 0 : stack.peek() + 1; // 左边界
                int width = right - left + 1;
                maxArea = Math.max(maxArea, curMaxHeight * width);
            }
        }
        return maxArea;
    }
}
```  

## 300. Longest Increasing Subsequence

```java
// DP, O(N^2)
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}

// O(NLogN) solution
// Binary Search
// 用一个数组，存储以下标+1为长度情况下的递增序列的最小尾值
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int len = 0;
        for (int num : nums) {
            // i, j 代表二分查找的lo, hi
            int i = 0;
            int j = len;
            // 二分法找到x在tail中的位置
            // 如果x大于所有tail的值，加到最后一个
            // 如果不是，用x值替代 符合tail[i] < x <= tail[i + 1]条件下，
            // tail[i + 1]的值
            while (i != j) {
                int m = (i + j) / 2;
                if (tails[m] < num) {
                    i = m + 1;
                } else {
                    j = m;
                }
            }
            tails[i] = num;
            // i == len意味着x大于所有tail的值，x被添加到了tail的末尾
            // 此时要更新len，用于下一轮的比较
            if (i == len) {
                len++;
            }
        }
        return len;
    }
}
```  

## 1002. Find Common Characters

```java
class Solution {
    public List<String> commonChars(String[] A) {
        List<String> res = new ArrayList<>();
        int[] count = new int[26];
        Arrays.fill(count, Integer.MAX_VALUE);
        for (String a : A) {
            int[] cnt = new int[26];
            for (char c : a.toCharArray()) {
                cnt[c - 'a']++;
            }
            for (int i = 0; i < 26; i++) {
                count[i] = Math.min(count[i], cnt[i]);
            }
        }
        for (char c = 'a'; c <= 'z'; c++) {
            while (count[c - 'a'] > 0) {
                res.add(String.valueOf(c));
                count[c - 'a']--;
            }
        }
        return res;
    }
}
```  

## 287. Find the Duplicate Number

```java
// Binary Search, O(NLogN)
class Solution {
    public int findDuplicate(int[] nums) {
        int lo = 1;
        int hi = nums.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int cnt = 0;
            for (int num : nums) {
                if (num <= mid) {
                    cnt++;
                }
            }
            if (cnt <= mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return hi;
    }
}
// Floyd Cycle, O(N) time
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];
        // find intersection
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        // find loop entrance
        int p1 = slow;
        int p2 = nums[0];
        while (p1 != p2) {
            p1 = nums[p1];
            p2 = nums[p2];
        }
        return p1;
    }
}
```  
