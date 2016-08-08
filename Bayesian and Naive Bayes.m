% i)Form and plot a data set Image from book consisting from 5000 points from ?1 and another 500 points from ?2.
function f = classifier(Sigma1,Sigma2)
mu1 = [0 2];
X1 = mvnrnd(mu1,Sigma1,5000); 
X1Class=[X1 zeros(size(X1,1),1)];
mu2 = [0 0];
X2 = mvnrnd(mu2,Sigma2,500);
X2Class=[X2 ones(size(X2,1),1)];
X=cat(1,X1Class,X2Class);

figure
idx1 = any(X(:,3)==0,2);
idx2=any(X(:,3)==1,2);
plot(X(idx1,1),X(idx1,2),'x',X(idx2,1),X(idx2,2),'o')
legend('Class1','Class2')
title('Plot of the points in the two classes');
total=size(X,1);

% ii)Assign each one of the points of Image from book to either ?1 or ?2, according to the Bayes decision rule, and plot the points with different colors, according to the class they are assigned to
% iv)Compute the error classification probability.
probOfw1=size(X1,1)/total;
probOfw2=size(X2,1)/total;
errorCount=0;
XBayesClassification=[];
for i=1:size(X)
  conditionalP1 = mvnpdf(X(i,1:2),mu1,Sigma1)*probOfw1;
  conditionalP2 = mvnpdf(X(i,1:2),mu2,Sigma2)*probOfw2;
  if conditionalP1>conditionalP2
      XBayesClassification=[XBayesClassification;X(i,1:2) 0];
      if X(i,3)==1
        errorCount=errorCount+1;
      end  
  else
      XBayesClassification=[XBayesClassification;X(i,1:2) 1];
      if X(i,3)==0 
        errorCount=errorCount +1;
      end
  end
end


errorBayes=[' The error in Bayes decision rule is ',num2str((errorCount/total)*100),'%'];
display(errorBayes)

figure
idx1 = any(XBayesClassification(:,3)==0,2);
idx2=any(XBayesClassification(:,3)==1,2);
plot(XBayesClassification(idx1,1),XBayesClassification(idx1,2),'x',XBayesClassification(idx2,1),XBayesClassification(idx2,2),'o')
legend('Class1','Class2')
title('Bayes decision rule');
% iv)Assign each one of the points of Image from book to either ?1 or ?2, according to the naive Bayes decision rule, and plot the points with different colors, according to the class they are assigned to.
% v)Compute the error classification probability, for the naive Bayes classifier.
features = X(:,1:2);
class = X(:,3);
NBModel = fitNaiveBayes(features,class);
label = predict(NBModel,X(:,1:2)); 
errorCountNaiveBayes=0;
for i=1:size(X);
    if label(i)~= X(i,3)
        errorCountNaiveBayes=errorCountNaiveBayes+1;
    end
end
errorBayes=[' The error in naive Bayes classifier is ',num2str((errorCountNaiveBayes/total)*100),'%'];
display(errorBayes)
NaiveBayes=[X(:,1:2) label];

figure
idx1 = any(NaiveBayes(:,3)==0,2);
idx2=any(NaiveBayes(:,3)==1,2);
plot(NaiveBayes(idx1,1),NaiveBayes(idx1,2),'x',NaiveBayes(idx2,1),NaiveBayes(idx2,2),'o')
legend('Class1','Class2')
title('Naive Bayes classification');
end