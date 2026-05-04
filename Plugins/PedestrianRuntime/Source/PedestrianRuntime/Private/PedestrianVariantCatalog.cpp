#include "PedestrianVariantCatalog.h"

const FPedVariantSpec* UPedestrianVariantCatalog::FindVariantById(FName VariantId) const
{
	if (VariantId.IsNone())
	{
		return nullptr;
	}

	const FString TrimmedName = VariantId.ToString().TrimStartAndEnd();
	if (TrimmedName.IsEmpty())
	{
		return nullptr;
	}

	const FName NormalizedId(*TrimmedName);
	return Variants.FindByPredicate(
		[NormalizedId](const FPedVariantSpec& Spec)
		{
			return Spec.VariantId == NormalizedId;
		});
}

bool UPedestrianVariantCatalog::FindVariantById(FName VariantId, FPedVariantSpec& OutSpec) const
{
	const FPedVariantSpec* FoundSpec = FindVariantById(VariantId);
	if (FoundSpec == nullptr)
	{
		return false;
	}

	OutSpec = *FoundSpec;
	return true;
}
