---
title: VSCodeでのjupyterとvimの共用
tags:
  - Vim
  - Jupyter
  - VSCode
private: false
updated_at: '2020-12-19T17:54:36+09:00'
id: b1db862cffbc0c6afdf6
organization_url_name: null
slide: false
ignorePublish: false
---
#拡張機能、jupyterとvimを使おうとしたが、
escキーが衝突する。jupyterではセルのエディットモードからコマンドモードに変わる機能が、vimではノーマルモードに変わる機能が割り当たっている。そのままではvimのコマンドが優先されて、jupyterのエディットモードから抜けられない。
そこでsetting.jsonに次のコードを追加する。

```javascript:setting.json
"vim.normalModeKeyBindings": [
        {
            "before": ["<Esc>"],
            "after":[],
            "commands":[
                {
                    "command":"notebook.cell.quitEdit",
                    "when":"vim.active && vim.mode == 'Normal'"
                }
            ]

        }
    ]

```
これでinsert-modeから2回Escを押すと、jupyterのコマンドモードに移行できる。
#こんなことを書きつつ
vimは全然わかっていない。すべてキーボードで完結したいと思って勉強中。
#参考にしたURL
https://github.com/VSCodeVim/Vim/issues/5238
