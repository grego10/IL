from rest_framework.response import Response
from rest_framework.decorators import api_view

from api.service.regression import predict_visitors_with_populatuion


@api_view(["GET"])
def getData(request):
    pop = request.query_params.get("pop")
    result = {"visitors": predict_visitors_with_populatuion(pop)}
    return Response(result)
