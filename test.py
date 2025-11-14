import random

class Detective:
    def __init__(self, name, psyche=12, intellect=14, physique=10, motorics=8):
        self.name = name
        self.psyche = psyche
        self.intellect = intellect
        self.physique = physique
        self.motorics = motorics

    def roll_check(self, skill, dc, narration, voices):
        """骰子检定 + 人格内心独白"""
        value = getattr(self, skill)
        roll = random.randint(1, 20)
        total = roll + (value - 10) // 2
        success = total >= dc

        print(f"\n🕵️ 场景: {narration}")
        print(f"🎲 检定: {skill.upper()} | 骰子 {roll} | 修正 {(value - 10)//2:+} | 总 {total} vs DC {dc}")

        # 内心人格对话
        for voice, line in voices.items():
            print(f"  🗨️ {voice}: {line}")

        if success:
            print("✅ 成功。你稳稳地掌控了局面。")
        else:
            print("❌ 失败。现实狠狠甩了你一巴掌。")
        return success

def main():
    hero = Detective("马洛（或许是另一个倒霉蛋）")

    print("=== 硬汉侦探 CRPG Demo ===")

    print("雨夜，城市的霓虹像溺水的鬼魂，在酒吧的破窗上打碎成片。")
    print("你推门进去，烟雾和廉价威士忌的气味扑面而来。")

    choice = input("\n酒保盯着你：'你要点什么？' \n[1] 用 PSYCHE（精神）强装冷酷 \n[2] 用 INTELLECT（智力）套话 \n> ")

    if choice == "1":
        hero.roll_check(
            "psyche",
            dc=13,
            narration="你死死盯着酒保，试图用目光压倒他。",
            voices={
                "PSYCHE": "别眨眼。狼不会对羊低头。",
                "INTELLECT": "真蠢，你看起来更像一条失眠的狗。",
            }
        )
    elif choice == "2":
        hero.roll_check(
            "intellect",
            dc=15,
            narration="你决定绕点弯子，从他嘴里撬出点东西。",
            voices={
                "INTELLECT": "措辞要精巧，就像解一道三流纵横字谜。",
                "PHYSIQUE": "少废话，直接一拳更快。",
            }
        )
    else:
        print("\n你沉默不语，只是点了一杯最便宜的酒。夜色比酒更苦。")

    print("\n=== Demo 结束，外面依旧是雨夜 ===")

if __name__ == "__main__":
    main()