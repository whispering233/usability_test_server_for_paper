import json

from django.shortcuts import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from app_predict.predict import SimilarityPredict
from app_predict.cos_predict import CosSimilarityPredict


# 实例化 predict 类
similarity_predict = SimilarityPredict()
cos_similarity_predict = CosSimilarityPredict()

# Create your views here.


# 定义需要实现的函数功能 或 渲染的页面
# request 是默认变量，是前端传过来的封装对象
@csrf_exempt
def predict(request):
    # http://127.0.0.1:8000/predict/%EF%BC%9Fsentence_pair=[%22%E5%96%9C%E6%AC%A2%E6%89%93%E7%AF%AE%E7%90%83%E7%9A%84%E7%94%B7%E7%94%9F%E5%96%9C%E6%AC%A2%E4%BB%80%E4%B9%88%E6%A0%B7%E7%9A%84%E5%A5%B3%E7%94%9F%22,%20%22%E7%88%B1%E6%89%93%E7%AF%AE%E7%90%83%E7%9A%84%E7%94%B7%E7%94%9F%E5%96%9C%E6%AC%A2%E4%BB%80%E4%B9%88%E6%A0%B7%E7%9A%84%E5%A5%B3%E7%94%9F%22]
    # print(request.method)
    # print(request.GET)
    # 不能通过 .xxx 的形式调用数据，毕竟不是类属性
    # 通过 get(key) 的方式进行获取数据
    # 就算获取到了数据，实际上也只是一个字符串
    sentence_pair = request.GET.get("sentence_pair")
    # print(sentence_pair)
    # print(type(sentence_pair))
    # sentence_pair = list(sentence_pair)
    # print(type(sentence_pair))
    # print(sentence_pair)
    # print(sentence_pair[0])
    # print(sentence_pair[1])

    # print("*" * 30)
    # sentence_pair = json.load(request.GET)
    # print(sentence_pair)

    temp_dict = request.GET.dict()
    # print(temp_dict)
    sentence_pair = temp_dict['sentence_pair']
    # print(sentence_pair)
    # print(type(sentence_pair))
    divide = sentence_pair.index(",")
    pair_list = [sentence_pair[2:divide-1], sentence_pair[divide+3:-2]]
    # print(pair_list)
    # print(pair_list[0])
    # print(pair_list[1])

    res = {"similarity": similarity_predict.predict(pair_list)}

    print(res)

    # HttpResponse 返回内容字符串
    # render 返回渲染的页面 return render(request, *.html, data_dict)
    # return redirect(url) 执行重定向
    # return HttpResponse(res)
    return JsonResponse(res)


@csrf_exempt
def predict_sequence(request):
    # print(request.encoding)
    # print(request.content_type)
    # print(request.method)
    data = request.body
    # print(data)
    # print(type(data))
    # print(data.decode('utf_8'))
    data = json.loads(data.decode('utf_8'))
    # print(data)
    # print(type(data))
    status = 201
    res = {
        "status": status
    }
    sentences_sequence = data['sentences_list']
    if len(sentences_sequence) < 2:
        return JsonResponse(res)
    predict = similarity_predict.predict_sequence(sentences_sequence)
    print(predict)
    predict = similarity_predict.computed_similarity(predict)
    print(predict)
    res['predict'] = predict
    res['status'] = 200
    return JsonResponse(res)


@csrf_exempt
def predict_one_sequence(request):

    data = request.body
    data = json.loads(data.decode('utf_8'))

    status = 201
    res = {
        "status": status
    }
    sentences_sequence = data['sentences_list']
    if len(sentences_sequence) < 2:
        return JsonResponse(res)
    _predict, similarity = similarity_predict.predict_one_sequence(sentences_sequence)
    print(_predict)
    print(similarity)
    res['predict'] = similarity
    res['status'] = 200
    return JsonResponse(res)

@csrf_exempt
def cos_predict_one_sequence(request):

    print("****************")

    data = request.body
    data = json.loads(data.decode('utf_8'))

    status = 201
    res = {
        "status": status
    }
    sentences_sequence = data['sentences_list']
    if len(sentences_sequence) < 2:
        return JsonResponse(res)
    _predict, similarity = cos_similarity_predict.predict_one_sequence(sentences_sequence)
    print(similarity)
    res['predict'] = similarity
    res['status'] = 200
    return JsonResponse(res)
