function [P_C, Ph_C, P_G, Ph_G, P_S, Ph_S, cc, gg, ss, sizeC, sizeG, sizeS,in1,in2] = generate_LORAKS_operators(N1, N2, R)
% function [P_C, Ph_C, P_G, Ph_G, P_S, Ph_S, cc, gg, ss, sizeC, sizeG, sizeS] = generate_LORAKS_operators(N1, N2, R)
% Justin Haldar 2/20/2014
% Inputs:
%     N1 x N2: The size of the fully-sampled k-space matrix
%     R:  The k-space radius used to construct LORAKS neighborhoods
% Outputs:
%     P_C, P_G, P_S: Operators to convert fully k-space data into the
%                    C, G, and S matrices, respectively
%     Ph_C, Ph_G, Ph_S: The adjoints of the P_C, P_G, and P_S operators,
%                       respectively
%     cc, gg, ss:  The diagonal entries of the Ph_C*P_C, Ph_G*P_G, Ph_S*P_S
%                  operators, respectively
%     sizeC, sizeG, sizeS: 2x1 vectors describing the sizes of the C, G,
%                          and S matrices, respectively
%
% This software is available from <a
% href="matlab:web('http://mr.usc.edu/download/LORAKS/')">
% http://mr.usc.edu/download/LORAKS/</a>.  As described on that page, use
% of this software (or its derivatives) in your own work requires that you
% cite the references listed at <a
% href="matlab:web('http://mr.usc.edu/download/LORAKS/')">
% http://mr.usc.edu/download/LORAKS/</a>.
%

if min(N1,N2 > 1)
    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);
    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);
    Ind = zeros(patchSize,(N1-2*R-even(N1))*(N2-2*R-even(N2)));
    IndG = zeros(patchSize,(N1-2*R)*(N2-2*R));
    Indp = zeros(patchSize,(N1-2*R-even(N1))*(N2-2*R-even(N2)));
    Indt = zeros(1,(N1-2*R)*(N2-2*R));
    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind(:,k) = sub2ind([N1,N2],i+in1,j+in2);
            Indp(:,k) = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);
        end
    end
    
    k = 0;
    for i = R+1:N1-R
        for j = R+1:N2-R
            k = k+1;
            IndG(:,k) = sub2ind([N1,N2],i+in1,j+in2);
            Indt(:,k) = sub2ind([N1,N2],-i    +2*ceil((N1-1)/2)+2,-j    +2*ceil((N2-1)/2)+2);
        end
    end
else
    in1 =[-R:R]';
    patchSize = numel(in1);
    Ind = zeros(patchSize,N1-2*R-even(N1));
    IndG = zeros(patchSize,N1-2*R);
    Indp = zeros(patchSize,N1-2*R-even(N1));
    Indt = zeros(1,N1-2*R);
    k = 0;
    for i = R+1+even(N1):N1-R
        k = k+1;
        Ind(:,k) = i+in1;
        Indp(:,k) = -i+in1+2*ceil((N1-1)/2)+2;
    end
    
    k = 0;
    for i = R+1:N1-R
        k = k+1;
        IndG(:,k) = i+in1;
        Indt(:,k) = -i    +2*ceil((N1-1)/2)+2;
    end
end

I = speye(N1*N2);
I = I(Ind(:),:);
Ip = I';

Ig = speye(N1*N2);
Ig = Ig(IndG(:),:);
Igp = Ig';

It = speye(N1*N2);
It = It(Indt(:),:);
Itp = It';

Ir = speye(N1*N2);
Ir = Ir(Indp(:),:);
Irp = Ir';


nPatch = size(I,1)/patchSize;
nPatchG = size(Ig,1)/patchSize;

P_C = @(x) reshape(I*(vect(x)),patchSize,nPatch);

Ph_C = @(x) Ip*vect(x);

P_G = @(x) [...
    reshape( Ig*vect(x(1:N1*N2)),patchSize,nPatchG),            reshape(Ig*vect(x(N1*N2+[1:N1*N2])),patchSize,nPatchG);...
    reshape(-Ig*vect(x(N1*N2+[1:N1*N2])),patchSize,nPatchG),    reshape(Ig*vect(x(1:N1*N2)),patchSize,nPatchG);...
    reshape(-It*vect(x(1:N1*N2)),1,nPatchG),                    reshape(It*vect(x(N1*N2+[1:N1*N2])),1,nPatchG);...
    ];
Ph_G = @(x) [...
    Igp*vect(x([1:patchSize],[1:nPatchG]))+Igp*vect(x(patchSize+[1:patchSize],nPatchG+[1:nPatchG]))-Itp*vect(x(2*patchSize+1,[1:nPatchG]));...
    Igp*vect(x([1:patchSize],nPatchG+[1:nPatchG]))-Igp*vect(x(patchSize+[1:patchSize],[1:nPatchG]))+Itp*vect(x(2*patchSize+1,nPatchG+[1:nPatchG]));...
    ];

P_S = @(x) [...
    reshape(I*vect(x(1:N1*N2))-Ir*vect(x(1:N1*N2)),patchSize,nPatch),  reshape(I*vect(x(N1*N2+[1:N1*N2]))+Ir*vect(x(N1*N2+[1:N1*N2])),patchSize,nPatch);...
    reshape(-I*vect(x(N1*N2+[1:N1*N2]))+Ir*vect(x(N1*N2+[1:N1*N2])),patchSize,nPatch), reshape(I*vect(x(1:N1*N2))+Ir*vect(x(1:N1*N2)),patchSize,nPatch);...
    ];
Ph_S = @(x) [...
    Ip*vect(x([1:patchSize],[1:nPatch]))-Irp*vect(x([1:patchSize],[1:nPatch]))+Ip*vect(x(patchSize+[1:patchSize],nPatch+[1:nPatch]))+Irp*vect(x(patchSize+[1:patchSize],nPatch+[1:nPatch]));...
    Ip*vect(x([1:patchSize],nPatch+[1:nPatch]))+Irp*vect(x([1:patchSize],nPatch+[1:nPatch]))-Ip*vect(x(patchSize+[1:patchSize],[1:nPatch]))+Irp*vect(x(patchSize+[1:patchSize],[1:nPatch]));...
    ];

ss = real(complexify(Ph_S(P_S(ones(2*N1*N2,1)))));
gg = real(complexify(Ph_G(P_G(ones(2*N1*N2,1)))));
cc = Ph_C(P_C(ones(N1*N2,1)));

sizeG = [patchSize*2+1,nPatchG*2];
sizeS = [2*patchSize,nPatch*2];
sizeC = [patchSize,nPatch];
return;

%%
function result = even(int)
result = not(rem(int,2));
return;

%%
function out = vect(in)
out=in(:);
return;

%%
function out = complexify(in)
out = complex(in(1:end/2,:,:,:,:,:,:,:,:),in(end/2+1:end,:,:,:,:,:,:,:,:));
return;