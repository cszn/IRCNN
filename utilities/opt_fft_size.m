function m = opt_fft_size(n)
% opt_fft_size.m
%
%   compute an optimal data length for Fourier transforms
%
%     written by Sunghyun Cho (sodomau@postech.ac.kr)
%
persistent opt_fft_size_LUT;

LUT_size = 4096;

if isempty(opt_fft_size_LUT)
    fprintf('generate opt_fft_size_LUT\n');
    opt_fft_size_LUT = zeros(LUT_size,1);

    e2 = 1;
    while e2 <= LUT_size
        e3 = e2;
        while e3 <= LUT_size
            e5 = e3;
            while e5 <= LUT_size
                e7 = e5;
                while e7 <= LUT_size
                    if e7 <= LUT_size
                        opt_fft_size_LUT(e7) = e7;
                    end
                    if e7*11 <= LUT_size
                        opt_fft_size_LUT(e7*11) = e7*11;
                    end
                    if e7*13 <= LUT_size
                        opt_fft_size_LUT(e7*13) = e7*13;
                    end
                    e7 = e7 * 7;
                end
                e5 = e5 * 5;
            end
            e3 = e3 * 3;
        end
        e2 = e2 * 2;
    end

    nn = 0;
    for i=LUT_size:-1:1
        if opt_fft_size_LUT(i) ~= 0
            nn = i;
        else
            opt_fft_size_LUT(i) = nn;
        end
    end
end

m = zeros(size(n));
for c = 1:numel(n)
    nn = n(c);
    if nn <= LUT_size
        m(c) = opt_fft_size_LUT(nn);
    else
        m(c) = -1;
    end
end
