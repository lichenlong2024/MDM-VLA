# 鐪熸満鎺ㄧ悊浣跨敤璇存槑

## 鐩爣

鎶?**PC1锛圧OS + MoveIt + 鍙岀浉鏈?+ Franka 鏈烘鑷傦級** 鍜?**AutoDL锛圙PU 鎺ㄧ悊鏈嶅姟锛?* 杩炶捣鏉ワ紝瀹炵幇鐪熸満闂幆鎺ㄧ悊锛?
1. PC1 閲囬泦涓よ矾 RGB 鍥惧儚鍜屽綋鍓嶆満姊拌噦鐘舵€?`state`
2. PC1 閫氳繃 SSH 闅ч亾璇锋眰 AutoDL 涓婄殑 `/act`
3. AutoDL 杩斿洖妯″瀷棰勬祴鐨?`action[0]`
4. PC1 灏嗗姩浣滆浆鎹㈡垚涓嬩竴姝ユ湯绔綅濮匡紝骞剁敤 MoveIt 鎵ц

褰撳墠鏈嶅姟绔悓鏃舵敮鎸佷袱绉嶆ā寮忥細

- Baseline: 鏅€?`L1RegressionActionHead`
- MoE: `ActionIDDiscriminator + StageMoEActionHead`锛屽嵆 Top-1 action-id 璺敱涓撳

---

## 鏂囦欢鍒嗗伐

### PC1 鏈湴

璐熻矗锛?
- ROS / MoveIt
- 鐩磋繛 Franka 鏈烘鑷?- 涓よ矾 RealSense 鍥惧儚閲囬泦
- 璇诲彇褰撳墠鏈鐘舵€?- 璋冪敤 AutoDL 鎺ㄧ悊鏈嶅姟
- 鎵ц妯″瀷杩斿洖鐨勫姩浣?
涓昏鏂囦欢锛?
- `client_to_VLA_Adapter.py`
- `Multi_P2P_MoveIt_Control.py`

### AutoDL 鏈嶅姟鍣?
璐熻矗锛?
- 鍔犺浇 VLA 妯″瀷
- 鍔犺浇 action head
- MoE 妯″紡涓嬪姞杞?action-id discriminator
- 浠?checkpoint 鐩綍璇诲彇 `dataset_statistics.json`
- 鎺ユ敹 `full_image / wrist_image / state / goal`
- 杩斿洖 `action`

涓昏鏂囦欢锛?
- `experiments/robot/server_deploy/franka_deploy_json.py`
- `experiments/robot/openvla_utils.py`
- `experiments/robot/robot_utils.py`
- 妯″瀷 checkpoint 鐩綍

---

## 1. 鍚姩 AutoDL 鎺ㄧ悊鏈嶅姟

### Baseline 妯″瀷

```bash
python experiments/robot/server_deploy/franka_deploy_json.py \
  --host 127.0.0.1 \
  --port 8000 \
  --pretrained_checkpoint <PATH_TO_MDM_VLA_CHECKPOINT> \
  --num_images_in_input 2 \
  --use_proprio true \
  --proprio_dim 8 \
  --use_moe_action_head True \
  --use_action_id_discriminator True
```

### MoE Top-1 Action-ID 妯″瀷

```bash
python experiments/robot/server_deploy/franka_deploy_json.py \
  --host 127.0.0.1 \
  --port 8000 \
  --pretrained_checkpoint <PATH_TO_MDM_VLA_CHECKPOINT> \
  --num_images_in_input 2 \
  --use_proprio true \
  --proprio_dim 8 \
  --use_pro_version true \
  --use_moe_action_head true \
  --use_action_id_discriminator true \
  --stage_definitions "0:0-2,1:3-10,2:11-17" \
  --num_action_ids 18
```

MoE checkpoint 鐩綍涓渶瑕佸寘鍚細

- `dataset_statistics.json`
- `action_head...checkpoint...`
- `action_id_discriminator...checkpoint...`
- `proprio_projector...checkpoint...`锛屽鏋滆缁冩椂浣跨敤浜?proprio

娉ㄦ剰锛歚find_checkpoint_file(...)` 鐩墠瑕佹眰鍚屼竴涓?checkpoint 鐩綍閲屾瘡绫荤粍浠跺彧鍖归厤鍒颁竴涓枃浠躲€傚鏋滃悓鐩綍涓嬫湁澶氫釜 `action_head` 鎴栧涓?`action_id_discriminator` checkpoint锛岄渶瑕佸厛鏁寸悊鐩綍锛屾垨淇敼鍔犺浇閫昏緫涓鸿嚜鍔ㄩ€夋嫨鏈€鏂般€?
---

## 2. 寤虹珛 SSH 闅ч亾

鍦?PC1 涓婃墽琛岋細

```bash
ssh -N -L 8000:127.0.0.1:8000 -p 12784 root@connect.westd.seetacloud.com
```

鍚箟锛?
- PC1 璁块棶 `127.0.0.1:8000`
- 瀹為檯璇锋眰浼氳杞彂鍒?AutoDL 鐨?`127.0.0.1:8000`
- AutoDL 鏈嶅姟绔彧闇€瑕佺洃鍚湰鏈哄湴鍧€鍗冲彲

---

## 3. 杩愯 PC1 鐪熸満瀹㈡埛绔?
```bash
python client_to_VLA_Adapter.py _trajectory_log_path:="<PATH_TO_REAL_ROBOT_TRAJECTORY_LOG>.json"
```

瀹㈡埛绔細鎵ц锛?
- 鍚姩涓よ矾鐩告満绾跨▼
- 灏嗘満姊拌噦绉诲姩鍒版暟鎹噰闆嗘椂鐨勫垵濮嬪叧鑺備綅濮?- 绛夊緟浜哄伐纭鍚庡紑濮嬫帹鐞?- 姣忔璇锋眰 AutoDL `/act`
- 鍙彇妯″瀷杩斿洖鐨勭涓€涓姩浣?`action[0]`
- 浣跨敤 MoveIt 鎵ц涓嬩竴姝ヤ綅濮?
---

## 璇锋眰涓庤繑鍥炴牸寮?
### PC1 鍙戠粰 AutoDL

```json
{
  "full_image": "<base64 png/jpg>",
  "wrist_image": "<base64 png/jpg>",
  "goal": "fold the side flap of the cardboard box",
  "state": [x, y, z, qx, qy, qz, qw, gripper]
}
```

### Baseline 杩斿洖

```json
{
  "action": [[dx, dy, dz, dr, dp, dyaw, gripper]]
}
```

### MoE 杩斿洖

```json
{
  "action": [[dx, dy, dz, dr, dp, dyaw, gripper]],
  "action_id_pred": 0
}
```

`action_id_pred` 鐢ㄤ簬纭鐪熸満鎺ㄧ悊鏃剁‘瀹炵粡杩囦簡 action-id discriminator 鍜?MoE 璺敱銆傚浜庡綋鍓?`libero_bend_action_id` 鏍囨敞锛?
- steps `0-34`: `action_id = 0`
- steps `35-74`: `action_id = 5`
- steps `75-end`: `action_id = 15`

瀵瑰簲褰撳墠闃舵鍒掑垎锛?
- `action_id 0` -> expert/stage `0`
- `action_id 5` -> expert/stage `1`
- `action_id 15` -> expert/stage `2`

---

## 宸插榻愰厤缃?
### 鍥惧儚

- 涓よ瑙掞細`full_image + wrist_image`
- 鏈嶅姟绔娇鐢ㄤ笌璁粌/浠跨湡涓€鑷寸殑 processor
- PC1 鏈湴涓嶅仛棰濆褰掍竴鍖?
### State

```python
state = [pos(3), quat(4), gripper(1)]
```

- 鎬荤淮搴︼細8
- 鍥涘厓鏁伴『搴忥細`xyzw`

### Action

```python
action = [dx, dy, dz, dr, dp, dyaw, gripper]
```

- 妯″瀷杈撳嚭褰㈢姸锛歚[N, 7]`
- 鐪熸満褰撳墠鍙娇鐢ㄧ涓€姝ワ細`action[0]`
- `dx, dy, dz` 鍙犲姞鍒板綋鍓嶆湯绔綅缃?- `dr, dp, dyaw` 浣滀负 RPY 澧為噺鍙犲姞鍒板綋鍓嶅Э鎬?
### 鎺у埗棰戠巼

- 褰撳墠鐪熸満寰幆棰戠巼锛氱害 `10Hz`

---

## 鍒濆浣嶅Э

鎺ㄧ悊鍓嶅繀椤诲厛鍥炲埌鏁版嵁閲囬泦鏃剁殑鍒濆浣嶅Э銆?
```python
ready_position = [0.33029, 0.01901, 0.59196]
ready_orientation = [0.939321, 0.342376, 0.0100039, 0.018824]
ready_q = [
    -9.61573316e-05,
    -6.04586784e-01,
    -6.79237522e-04,
    -2.39317289e+00,
    -2.90650372e-02,
    1.82254346e+00,
    9.19703274e-02,
]
```

瀹㈡埛绔祦绋嬪簲涓猴細

1. 鍚姩鐩告満绾跨▼
2. 绉诲姩鍒?`ready_q`
3. 浜哄伐纭鐜绋冲畾
4. 寮€濮嬭姹?AutoDL 鎺ㄧ悊

---

## 鍚姩椤哄簭

1. AutoDL 涓婂惎鍔?`franka_deploy_json.py`
2. PC1 涓婂缓绔?SSH 闅ч亾
3. PC1 涓婅繍琛?`client_to_VLA_Adapter.py`
4. 瑙傚療鏈嶅姟绔繑鍥炵殑 `action[0]`
5. MoE 妯″紡涓嬪悓姝ヨ瀵?`action_id_pred`
6. 纭鍔ㄤ綔灏哄害姝ｅ父鍚庡啀杩炵画鎵ц

---

## MoE 妯″紡妫€鏌ラ」

濡傛灉浣犳兂纭鐪熸満鎺ㄧ悊娌℃湁閫€鍥?baseline锛岄噸鐐圭湅锛?
- 鍚姩鍛戒护閲?`--use_moe_action_head true`
- 鍚姩鍛戒护閲?`--use_action_id_discriminator true`
- 杩斿洖 JSON 涓嚭鐜?`action_id_pred`
- `action_id_pred` 鍦ㄤ换鍔¤繃绋嬩腑涓嶆槸姘歌繙鍥哄畾涓哄悓涓€涓€?- checkpoint 鐩綍涓‘瀹炴湁 `action_head` 鍜?`action_id_discriminator` 鏉冮噸

濡傛灉娌℃湁 `action_id_pred`锛岃鏄庡綋鍓嶈姹備粛鍦?baseline 璺緞涓娿€?
---

## 甯歌闂

### 1. 鎵句笉鍒?checkpoint

妫€鏌?`--pretrained_checkpoint` 鏄惁鎸囧悜涓€涓洰褰曪紝鑰屼笉鏄崟涓?`.pt` 鏂囦欢銆?
鐩綍涓嬪簲鏈夛細

- `dataset_statistics.json`
- `action_head...checkpoint...`
- `proprio_projector...checkpoint...`
- MoE 妯″紡杩橀渶瑕?`action_id_discriminator...checkpoint...`

### 2. action 灏哄害杩囧ぇ

鍏堝湪 PC1 瀹㈡埛绔晶鍔犻檺骞咃細

- 浣嶇疆澧為噺闄愬箙
- 濮挎€佸閲忛檺骞?- gripper 鍗曠嫭澶勭悊

### 3. 寤惰繜杈冮珮

浼樺厛妫€鏌ワ細

- 鍥惧儚鏄惁浠?PNG 浼犺緭
- SSH 闅ч亾寤惰繜
- AutoDL 棣栨 CUDA / TensorFlow JIT 鍒濆鍖?
鍙互鑰冭檻鎶婂浘鍍忔敼鎴?JPEG 鍚庡啀 base64銆?
### 4. MoE 棰勬祴 action-id 涓嶅彉鍖?
浼樺厛妫€鏌ワ細

- `action_id_discriminator` 鏄惁鎴愬姛鍔犺浇
- `use_action_id_discriminator` 鏄惁涓?true
- `use_moe_action_head` 鏄惁涓?true
- 鐪熸満鍥惧儚鍜岃缁冨浘鍍忚瑙掓槸鍚︿竴鑷?- `state` 缁村害鍜岄『搴忔槸鍚︿竴鑷?
---

## 褰撳墠鐘舵€?
- `franka_deploy_json.py` 鏀寔 baseline 鍜?MoE 鍙屾ā寮?- MoE 妯″紡浼氳繑鍥?`action_id_pred`
- 鏈嶅姟绔細鑷姩浠?checkpoint 鐩綍璇诲彇 `dataset_statistics.json`
- PC1 閫氳繃 SSH 绔彛杞彂璁块棶 AutoDL
- 鐪熸満瀹㈡埛绔粯璁よ姹?`127.0.0.1:8000/act`


