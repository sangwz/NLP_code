'''
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
RuntimeError: exp_vml_cpu not implemented for 'Long'
错误原因：数据格式不匹配

'''
# 原代码
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
# 修改,将0处改为浮点数
div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

# 然后报错
# RuntimeError: expected device cpu and dtype Float but got device cpu and dtype Long
# 原代码
position = torch.arange(0, max_len).unsqueeze(1)
# 修改，将0改为0.，浮点数
position = torch.arange(0., max_len).unsqueeze(1)