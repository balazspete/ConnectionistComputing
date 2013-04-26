# Balazs Pete - 09771417

require_relative 'MLP.rb'

epochs = 3000
learning_rate = 0.3
data = [
    {input: [0, 0], output: [0]},
    {input: [0, 1], output: [1]},
    {input: [1, 0], output: [1]},
    {input: [1, 1], output: [0]}
]

nn = MLP.new 2, 12, 1

(1..epochs).each do |epoch|
  error = 0
  data.each do |d|
    nn.forward d[:input]
    error += nn.backwards d[:output]
    nn.update_weights learning_rate
  end

  puts "Error at epoch #{epoch} is #{error}"
end

nn.forward [-1,-1]
p nn.get_output
nn.forward [-1,1]
p nn.get_output
nn.forward [1,-1]
p nn.get_output
nn.forward [1,1]
p nn.get_output

#p nn.output

