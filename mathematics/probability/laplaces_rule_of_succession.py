'''
Ideas from: https://www.youtube.com/watch?v=8idr1WZ1A7Q&t=327s
They got their ideas from: https://www.johndcook.com/blog/2011/09/27/bayesian-amazon/
Say you have the following Amazon Ratings for the same product near
the same price.  Which seller should you choose?
100% with 10 ratings
96% with 50 ratings
93% with 200 ratings
Laplaces Rule of Succession gives you a way to calculate this
You just pretend to add 1 positive review and 1 negative review
(2 reviews total), and then recacalcuate the rating
'''


def laplace_rule_of_succession(global_rating, max_rating,
                               total_number_of_reviews):
    percent_positive = global_rating/max_rating
    print(f"Percent Positive {percent_positive * 100}")
    # Now, let's pretend that we had two more reviews,
    # one positive and one negative
    num_positive = percent_positive * total_number_of_reviews
    print(f"num positive {num_positive}")
    num_positive += 1
    total_number_of_reviews += 2
    real_rating = num_positive / total_number_of_reviews
    print(f"Real Rating {real_rating}")
    print()


laplace_rule_of_succession(5, 5, 10)
laplace_rule_of_succession((48/50) * 5, 5, 50)
laplace_rule_of_succession((186/200) * 5, 5, 200)
