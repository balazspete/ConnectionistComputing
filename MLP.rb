# Balazs Pete - 09771417

# A multi layer perceptron
class MLP
  attr_accessor :output
  def initialize ni, nh, no
    @input = Array.new(ni){MLP::Node.new}
    @hidden = Array.new(nh){MLP::Neuron.new(@input)}
    @output = Array.new(no){MLP::Neuron.new(@hidden)}

  end

  def forward input
    @input.each_index do |i|
      @input[i].value = input[i]
    end
    @hidden.each do |n|
      n.calc
    end
    @output.each do |n|
      n.calc
    end
    @output
  end

  def backwards target
    _error = 0

    # Foreach output node, calculate the error
    @output.each_index do |i|
      o = @output[i]
      o.error = o.value * (1-o.value) * (target[i] - o.value)
      _error += o.error

      # Foreach weight of the node, calculate change
      o.weights.each_index do |wi|
        o.change[wi] = o.error * @hidden[wi].value#o.weights[wi]
      end
    end

    # Foreach hidden node, calculate error
    @hidden.each_index do |i|
      h = @hidden[i]

      # error = "value of node" * (1-"value of node")* "sum of the errors of the connected weights"
      h.error = h.value * (1-h.value) * calc_hidden_node_weights_errors(i)

      # Foreach weight of the node, calculate change
      h.weights.each_index do |wi|
        # Change = "error of node" * "input value"
        h.change[wi] = h.error * @input[wi].value
      end
    end
    return _error
  end

  def update_weights learning_rate
    update_helper @output, learning_rate
    update_helper @hidden, learning_rate
  end

  def get_output
    arr = []
    @output.each do |o|
      arr.push o.value
    end
    arr
  end

  private
  def calc_hidden_node_weights_errors index
    sum = 0
    @output.each do |o|
      sum += o.error * o.weights[index]
    end
    sum
  end

  def update_helper array, learning_rate
    array.each do |n|
      n.weights.each_index do |i|
        n.weights[i] += n.change[i] * learning_rate
        n.change[i] = 0
      end
    end
  end

end

# A generic node of a MLP
class MLP::Node
  attr_accessor :value

  def initialize
    @value = 0
  end

end

# An extension of a node, containing a set of weights
class MLP::Neuron < MLP::Node
  attr_accessor :activation, :value, :weights, :change, :error

  def initialize input
    @activation, @value, @error = 0, 0, 0
    @input = input
    @weights = Array.new(input.size){rand()}
    @change = Array.new(input.size){0}
  end

  def calc sigmoid=true
    temp = 0
    @weights.each_index do |index|
      temp += @weights[index] * @input[index].value
    end
    @value = Math.tanh(temp).abs
  end
end

