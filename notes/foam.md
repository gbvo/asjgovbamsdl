在 GitHub 上新建文件夹很简单，不需要使用命令行，直接在网页上就能创建。

### 方法一：在 GitHub 网页上创建文件夹（推荐）

1. 打开你的 GitHub 仓库（Repository）。
2. 点击右上方的 Add file → Create new file。
3. 在文件名输入框中输入：
   ```
   my-folder/README.md
   ```
   其中：
   - `my-folder` 是你要创建的文件夹名称。
   - `/` 表示创建文件夹。
   - `README.md` 是文件夹里的文件。
4. 在下方文件编辑区域随便输入一点内容。
5. 点击右上方的 Commit changes，确认提交。

&#x20;完成！ 你的 GitHub 仓库中就会出现 `my-folder` 文件夹。

### 方法二：一次创建多级文件夹

例如，你想创建：

```
project/
  └── src/
      └── components/
          └── index.js
```

直接在新建文件页面输入：

```
project/src/components/index.js
```

提交后，GitHub 就会自动创建这些文件夹。

### 注意

GitHub 不能直接创建完全空的文件夹，因为 Git 不追踪空目录。

如果只想创建一个空文件夹，可以在里面放一个 `.gitkeep` 文件，例如：

```
images/.gitkeep
```

这样就能保留 `images` 文件夹。

如果你愿意，我也可以教你如何在 GitHub 网页上创建文件夹并上传文件到里面。
