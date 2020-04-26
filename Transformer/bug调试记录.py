"""
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

'''
RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.FloatTensor instead (while checking arguments for embedding)
'''
linux下没有此问题

'''
Process finished with exit code -1073741676 (0xC0000094)
'''
windows下运行出错，linux下没有出现此问题

'''
RuntimeError: bool value of Tensor with more than one value is ambiguous
'''

"""