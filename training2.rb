# Balazs Pete - 09771417

require_relative 'MLP.rb'

# Convert a character to its unary array representation
def char_to_unary_array char
  index = char.downcase[0].ord - 'a'.ord
  result = Array.new 26, 0
  result[index] = 1
  return result
end

def unary_array_to_char array
  index = array.index(1)
  index == nil ? 'a' : (index + 'a'.ord).chr
end

def transform_input input
  l = input.chomp.split(/,/)
  return l[1...l.size].map {|x| x.to_i}
end

nn = MLP.new 16, 10, 26
epochs = 20
learning_rate = 0.003



training_data = []
file = File.new("training-set.data", "r")
while (line = file.gets)
    t = Array.new(26, 0)
    training_data.push(
        input: transform_input(line),
        output: char_to_unary_array(line[0]))
end
file.close

testing_data = []
file = File.new("testing-set.data", "r")
while (line = file.gets)
    t = Array.new(26, 0)
    testing_data.push(
        input: transform_input(line),
        output: line[0])
end
file.close



(1..epochs).each do |epoch|
  error = 0
  training_data.each do |d|
    nn.forward d[:input]
    error += nn.backwards d[:output]
    nn.update_weights learning_rate
  end

  puts "Error at epoch #{epoch} is #{error}"
end

successful = 0
testing_data.each do |d|
  char = unary_array_to_char nn.forward(d[:input]).map{|x| x.value}
  if char == d[:output].downcase
    successful += 1
  end
end
puts "\nPerformance: #{successful.to_f/testing_data.size}"
