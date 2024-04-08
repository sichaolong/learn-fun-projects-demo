import codecs
import csv



def calculate_actual_remaining_hour(csvFilePath):
    with codecs.open(csvFilePath, encoding='utf-8-sig') as f:
        story_point_sum = 0.0
        actual_remaining_hour_sum = 0.0
        for row in csv.DictReader(f, skipinitialspace=True):
            print("debug log: {}".format(row))
            cur_task_name = row['标题']
            # Story Point
            cur_story_point = 0.0
            # 剩余工时
            cur_remaining_hour = 0.0
            # 实际工时
            cur_actual_hour = 0.0
            # 实际剩余工时
            # cur_actual_remaining_hour = 0.0

            # 1、计算【Story Point】
            if len(row['Story Point']) != 0:
                cur_story_point = float(row['Story Point'])
            # sum
            story_point_sum += cur_story_point

            # 2、计算每行的【实际剩余工时】，分为下面情况
            # 2.1、【剩余工时】不为空，直接作为【实际剩余工时】
            # 2.2、【剩余工时】为空，需要使用 Story Point * 8 - 【实际工时】
            if len(row['实际工时']) != 0:
                cur_actual_hour = float(row['实际工时'].replace('小时', ''))

            if len(row['剩余工时']) != 0:
                cur_remaining_hour = float(row['剩余工时'])
                cur_actual_remaining_hour = cur_remaining_hour
            else:
                cur_actual_remaining_hour = cur_story_point * 8 - cur_actual_hour
            # sum
            actual_remaining_hour_sum += cur_actual_remaining_hour

            print("debug log: 当前任务：{},【Story Point】：{},【实际工时】：{},【剩余工时】：{},【实际剩余工时】：{}\n".format(
                cur_task_name,
                cur_story_point,
                cur_actual_hour,
                cur_remaining_hour,
                cur_actual_remaining_hour))

        print("总【Story Point】工时：{} 天,{} 小时\n".format(story_point_sum, 8 * story_point_sum))
        print("总【实际剩余工时】工时：{} 小时\n".format(actual_remaining_hour_sum))

    f.close()


if __name__ == '__main__':
    # csvFilePath = "/Users/sichaolong/Documents/my_projects/my_pycharm_projects/hours-burndown-chart-calculation-script/csvfiles/【Alpha-QBM】任务信息表_20240108.csv"
    # csvFilePath = "./csvfiles/【Alpha-QBM】任务信息表_20240108.csv"
    csvFilePath = input("****** 欢迎使用【工时燃尽图自动计算】脚本 auth:scl *******\n"
                        "=====> 请输入csv文件（必须包含表头【标题、Story Point、剩余工时、实际工时】）的路径：")
    calculate_actual_remaining_hour(csvFilePath)
