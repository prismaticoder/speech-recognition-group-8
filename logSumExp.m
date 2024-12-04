function result = logSumExp(logX, logY)
    % Computes log(exp(logX) + exp(logY)) in a numerically stable way using
    % Log-Sum-Exp Trick
    if logX == -inf
        result = logY;
    elseif logY == -inf
        result = logX;
    else
        maxLog = max(logX, logY);
        result = maxLog + log(exp(logX - maxLog) + exp(logY - maxLog));
    end
end
