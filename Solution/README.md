# Questions

## 621. Task Schedular

```java
public int leastInterval(char[] tasks, int n) {
    // 统计每个字母的出现频率
    int[] counter = new int[26];
    int max = 0; // 最高频的次数
    int maxCnt = 0; // 同时出现最高频的个数
    for (char task : tasks) {
        int index = task - 'A';
        counter[index]++;
        if (max == counter[index]) {
            maxCnt++;
        } else if (max < counter[index]) {
            max = counter[index];
            maxCnt = 1;
        }
    }
    // 中间的间隔
    int partCnt = max - 1;
    // 间隔的长度
    int partLength = n - (maxCnt - 1);
    // 间隔总数
    int emptySlots = partLength * partCnt;
    // 还需要分配的任务个数
    int otherTasks = tasks.length - max * maxCnt;
    // 所需空闲数
    int idles = Math.max(0, emptySlots - otherTasks);
    // 返回任务长度加上额外的空闲数
    return tasks.length + idles;
}
```  

## 348. Design Tic-Tac-Toe

```java
// 思路，建立一个二维数组，每进行一次判断是否胜利
class TicTacToe {

    int[][] board;
    /** Initialize your data structure here. */
    public TicTacToe(int n) {
        board = new int[n][n];
    }
`
    /** Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins. */
    public int move(int row, int col, int player) {
        if (board[row][col] == 0) {
            board[row][col] = player;
        } else {
            return 0;
        }
        if (row == col && checkDiagonal(player)) {
            return player;
        }
        if (row + col == board.length - 1 && checkRevDiagonal(player)) {
            return player;
        }
        if (checkRow(row, player) || checkCol(col, player)) {
            return player;
        }
        return 0;
    }

    private boolean checkDiagonal(int p) {
        for (int i = 0; i < board.length; i++) {
            if (board[i][i] != p) {
                return false;
            }
        }
        return true;
    }

    private boolean checkRevDiagonal(int p) {
        for (int i = 0; i < board.length; i++) {
            if (board[i][board.length - 1 - i] != p) {
                return false;
            }
        }
        return true;
    }

    private boolean checkRow(int r, int p) {
        for (int i = 0; i < board.length; i++) {
            if (board[r][i] != p) {
                return false;
            }
        }
        return true;
    }

    private boolean checkCol(int c, int p) {
        for (int i = 0; i < board.length; i++) {
            if (board[i][c] != p) {
                return false;
            }
        }
        return true;
    }
}
```  

## 362. Design Hit Counter

输入的timestamp是单位为秒的时间, 5分钟也就是timestamp 300，使用一个队列来记录

```java
class HitCounter {

    Queue<Integer> q;
    /** Initialize your data structure here. */
    public HitCounter() {
        q = new LinkedList<>();
    }

    /** Record a hit.
        @param timestamp - The current timestamp (in seconds granularity). */
    public void hit(int timestamp) {
        q.offer(timestamp);
    }

    /** Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity). */
    public int getHits(int timestamp) {
        while (!q.isEmpty() && timestamp - q.peek() >= 300) {
            q.poll();
        }
        return q.size();
    }
}
```  

## 539. Minimum Time Difference

Bucket Sort

```java
public int findMinDifference(List<String> timePoints) {
    boolean[] time = new boolean[1440];
    for (String points : timePoints) {
        String[] t = points.split(":");
        int hour = Integer.parseInt(t[0]);
        int min = Integer.parseInt(t[1]);
        if (time[hour * 60 + min]) { // duplicate time
            return 0;
        }
        time[hour * 60 + min] = true;
    }
    int min = 1440;
    int prev = 0;
    int first = Integer.MAX_VALUE;
    int last = Integer.MIN_VALUE;
    for (int i = 0; i < 1440; i++) {
        if (time[i]) {
            if (first != Integer.MAX_VALUE) { // 排除第一个找到的时间，无法相减
                min = Math.min(min, i - prev);
            }
            first = Math.min(first, i); // 找到最小时间
            last = Math.max(last, i); // 找到最大时间
            prev = i;
        }
    }
    // corner case
    min = Math.min(min, (1440 - last + first)); // 取反向时间比较
    return min;
}
```  

## 393. UTF8 Validation

知识点 Bit Manipulation [UTF8](https://www.fileformat.info/info/unicode/utf8.htm)

```java
public boolean validUtf8(int[] data) {
    int cnt = 0;
    for (int d : data) {
        if (cnt == 0) {
            if (d >> 5 == 0b110) { // 2 byte
                cnt = 1;
            } else if (d >> 4 == 0b1110) { // 3 byte
                cnt = 2;
            } else if (d >> 3 == 0b11110) { // 4 byte
                cnt = 3;
            } else if (d >> 7 != 0) { // 1 byte and not start with 0
                return false;
            }
        } else {
            if (d >> 6 != 0b10) { // check remaining byte
                return false;
            }
            cnt--;
        }
    }
    return cnt == 0; // only return true when no byte remaining.
}
```  

## 193. Valid Phone Numbers

Bash，任意xxx-开头或者(xxx) 开头

```bash
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt
```  

## 1114. Print in Order

```java
// concurrency problem
// use Semaphore
import java.util.concurrent.*;

class Foo {

    Semaphore run2;
    Semaphore run3;

    public Foo() {
        run2 = new Semaphore(0);
        run3 = new Semaphore(0);
    }

    public void first(Runnable printFirst) throws InterruptedException {
        printFirst.run();
        run2.release();
    }

    public void second(Runnable printSecond) throws InterruptedException {
        run2.acquire();
        printSecond.run();
        run3.release();
    }

    public void third(Runnable printThird) throws InterruptedException {
        run3.acquire();
        printThird.run();
    }
}

// Use CountDownLatch
import java.util.concurrent.*;

class Foo {

    CountDownLatch run2;
    CountDownLatch run3;
    public Foo() {
        run2 = new CountDownLatch(1);
        run3 = new CountDownLatch(1);
    }

    public void first(Runnable printFirst) throws InterruptedException {
        printFirst.run();
        run2.countDown();
    }

    public void second(Runnable printSecond) throws InterruptedException {
        run2.await();
        printSecond.run();
        run3.countDown();
    }

    public void third(Runnable printThird) throws InterruptedException {
        run3.await();
        printThird.run();
    }
}
```
